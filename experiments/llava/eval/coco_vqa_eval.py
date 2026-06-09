"""COCO VQA evaluation entry-point for LLaVA + Fast Nyström Attention.

This script loads a LLaVA-NeXT checkpoint that has been instrumented with
Fast Nyström Attention (FNA) layers and runs inference on the COCO VQA
validation split. It produces:

* ``predictions.jsonl`` – detailed per-question generations.
* ``submission.json`` – minimal list of ``{"question_id", "answer"}`` for
  use with the official VQA evaluation server if desired.
* ``metrics.json`` – locally computed VQA accuracy using the public metric.

Example usage::

    python experiments/llava/eval/coco_vqa_eval.py \
        --model-id llava-hf/llava-v1.6-vicuna-7b-hf \
        --questions-json /path/to/v2_OpenEnded_mscoco_val2014_questions.json \
        --annotations-json /path/to/v2_mscoco_val2014_annotations.json \
        --images-root /path/to/coco/val2014 \
        --output-dir outputs/llava_fna_eval \
        --fna-layer-range 18:32 --fna-num-sample 256

The script deliberately keeps side effects (file IO, generation, scoring)
encapsulated so it can be imported from a notebook if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import string
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple
import sys

import torch
from PIL import Image
from tqdm import tqdm

# Ensure the project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GenerationRecord:
    """Structured representation of a single model generation."""

    question_id: int
    image_id: int
    question: str
    answer: str
    raw_completion: str
    latency_s: float
    num_starting_tokens: int
    num_generated_tokens: int
    generation_latency_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Aggregate stats that are persisted to ``metrics.json``."""

    num_questions: int
    num_predicted: int
    accuracy: Optional[float]
    avg_latency_s: Optional[float]
    median_latency_s: Optional[float]
    avg_generated_tokens: Optional[float]
    avg_tokens_per_s: Optional[float] = None
    avg_generation_latency_s: Optional[float] = None
    avg_generation_tokens_per_s: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text normalization helpers (VQA style)
# ---------------------------------------------------------------------------


ARTICLES = {"a", "an", "the"}
PUNCTUATION = string.punctuation

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "lets": "let's",
    "mightve": "might've",
    "mightnt": "mightn't",
    "mustve": "must've",
    "mustnt": "mustn't",
    "neednt": "needn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodyd": "somebody'd",
    "somebodys": "somebody's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whats": "what's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

DIGIT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _strip_punctuation(text: str) -> str:
    table = str.maketrans({ch: " " for ch in PUNCTUATION})
    return text.translate(table)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = _strip_punctuation(text)
    tokens: List[str] = []
    for token in text.split():
        token = DIGIT_WORDS.get(token, token)
        token = CONTRACTIONS.get(token, token)
        if token in ARTICLES:
            continue
        tokens.append(token)
    return " ".join(tokens)


def vqa_score(prediction: str, answers: Sequence[str]) -> float:
    if not prediction or not answers:
        return 0.0
    normalized_pred = normalize_answer(prediction)
    normalized_answers = [normalize_answer(ans) for ans in answers]
    match_count = sum(1 for ans in normalized_answers if ans == normalized_pred)
    return min(1.0, match_count / 3.0)


# ---------------------------------------------------------------------------
# Annotation store and aggregation
# ---------------------------------------------------------------------------


class CocoVqaAnnotations:
    """Loads ground-truth annotations and exposes fast lookup by question."""

    def __init__(self, annotation_path: Optional[Path]):
        self._answers: Dict[int, List[str]] = {}
        if annotation_path is None:
            logging.warning("Annotation file not provided – accuracy will be skipped.")
            return

        with annotation_path.open("r") as fp:
            payload = json.load(fp)

        annotations = payload.get("annotations", [])
        for ann in annotations:
            question_id = int(ann["question_id"])
            answers = [entry["answer"] for entry in ann.get("answers", [])]
            self._answers[question_id] = answers
        logging.info("Loaded %d ground-truth annotations from %s", len(self._answers), annotation_path)

    def answers_for(self, question_id: int) -> Optional[Sequence[str]]:
        return self._answers.get(question_id)

    def is_available(self) -> bool:
        return bool(self._answers)


def summarize_accuracy(predictions: Sequence[GenerationRecord], annotations: CocoVqaAnnotations) -> Optional[float]:
    if not annotations.is_available():
        return None
    scores: List[float] = []
    for record in predictions:
        gt_answers = annotations.answers_for(record.question_id)
        if not gt_answers:
            continue
        scores.append(vqa_score(record.answer, gt_answers))
    if not scores:
        return None
    return 100.0 * float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# BERTScore helpers
# ---------------------------------------------------------------------------


def select_reference_answer(answers: Sequence[str], strategy: str) -> str:
    if not answers:
        raise ValueError("select_reference_answer received no answers")
    if strategy == "first":
        return answers[0]
    if strategy == "majority":
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
    if strategy == "concat":
        unique_answers = list(dict.fromkeys(answers))
        return " \n".join(unique_answers)
    raise ValueError(f"Unknown reference strategy '{strategy}'")


def load_reference_answers(path: Optional[Path]) -> Dict[int, str]:
    references: Dict[int, str] = {}
    if path is None:
        return references

    raw_payload = path.read_text()
    try:
        payload = json.loads(raw_payload)
        entries: Sequence[Dict[str, object]]
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            entries = payload["data"]  # type: ignore[index]
        else:
            raise ValueError("Reference JSON must be a list or contain a 'data' list")
    except json.JSONDecodeError:
        entries = []
        for line in raw_payload.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    skipped = 0
    for entry in entries:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        if "question_id" not in entry or "answer" not in entry:
            skipped += 1
            continue
        try:
            question_id = int(entry["question_id"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        answer_text = str(entry.get("answer", "")).strip()
        if not answer_text:
            skipped += 1
            continue
        references[question_id] = answer_text

    logging.info(
        "Loaded %d external reference answers from %s (skipped %d malformed entries)",
        len(references),
        path,
        skipped,
    )
    return references


def compute_bertscore(
    predictions: Sequence[GenerationRecord],
    annotations: CocoVqaAnnotations,
    output_path: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, float]]:
    reference_answers = load_reference_answers(args.bertscore_reference_json)
    using_external_refs = bool(reference_answers)

    if (not using_external_refs) and (not annotations.is_available()):
        logging.warning("Skipping BERTScore because annotations were not provided.")
        return None

    examples: List[Tuple[GenerationRecord, str]] = []
    for record in predictions:
        if using_external_refs:
            reference = reference_answers.get(record.question_id)
            if not reference:
                continue
        else:
            answers = annotations.answers_for(record.question_id)
            if not answers:
                continue
            reference = select_reference_answer(answers, args.bertscore_reference_strategy)
        examples.append((record, reference))

    if not examples:
        logging.warning("No overlapping predictions and references for BERTScore computation.")
        return None

    try:
        import evaluate  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "BERTScore evaluation requires the 'evaluate' package. Install it via 'pip install evaluate bert-score'."
        ) from exc

    reference_desc = (
        f"{len(reference_answers)} external references"
        if using_external_refs
        else "ground-truth annotations"
    )
    logging.info(
        "Computing BERTScore on %d predictions using %s with %s",
        len(examples),
        args.bertscore_model_type,
        reference_desc,
    )

    bertscore = evaluate.load("bertscore")
    compute_kwargs = {
        "predictions": [record.answer for record, _ in examples],
        "references": [reference for _, reference in examples],
        "lang": args.bertscore_lang,
        "model_type": args.bertscore_model_type,
        "batch_size": args.bertscore_batch_size,
    }
    if args.bertscore_rescale:
        compute_kwargs["rescale_with_baseline"] = True
    device = args.bertscore_device or args.device
    if device:
        compute_kwargs["device"] = device

    scores = bertscore.compute(**compute_kwargs)
    precisions = [float(value) for value in scores["precision"]]
    recalls = [float(value) for value in scores["recall"]]
    f1s = [float(value) for value in scores["f1"]]

    with output_path.open("w") as fp:
        for (record, reference), precision, recall, f1 in zip(examples, precisions, recalls, f1s):
            fp.write(
                json.dumps(
                    {
                        "question_id": record.question_id,
                        "prediction": record.answer,
                        "reference": reference,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
                + "\n"
            )

    summary = {
        "precision": float(sum(precisions) / len(precisions)),
        "recall": float(sum(recalls) / len(recalls)),
        "f1": float(sum(f1s) / len(f1s)),
    }
    logging.info("BERTScore summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]


def parse_layer_selection(layer_range: str, explicit: Optional[Sequence[int]], disabled: bool) -> List[int]:
    if disabled:
        return []
    if explicit:
        return sorted(set(int(layer) for layer in explicit))
    if not layer_range:
        return []
    try:
        start_str, end_str = layer_range.split(":", maxsplit=1)
        start, end = int(start_str), int(end_str)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("--fna-layer-range must be formatted as start:end") from exc
    if end <= start:
        raise ValueError("--fna-layer-range end must be > start")
    return list(range(start, end))


def load_questions(question_path: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    with question_path.open("r") as fp:
        payload = json.load(fp)
    questions = payload.get("questions", [])
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError(f"No questions found in {question_path}")
    return questions


def image_path(images_root: Path, image_id: int, split: str) -> Path:
    filename = f"COCO_{split}_{image_id:012d}.jpg"
    candidate_root = images_root / split
    candidate = candidate_root / filename if candidate_root.exists() else images_root / filename
    if not candidate.exists():
        raise FileNotFoundError(f"Image not found: {candidate}")
    return candidate


def prepare_prompt(
    processor: LlavaNextProcessor,
    question: str,
    system_prompt: Optional[str],
    answer_guidance: Optional[str],
) -> str:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    user_text = question
    if answer_guidance:
        user_text = f"{question}\n\n{answer_guidance.strip()}"
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    )
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def move_batch_to_device(batch: MutableMapping[str, torch.Tensor], device: Optional[str], dtype: torch.dtype) -> MutableMapping[str, torch.Tensor]:
    if device is None:
        return batch
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device=device)
            if value.dtype in {torch.float32, torch.float16, torch.bfloat16}:
                batch[key] = batch[key].to(dtype)
    return batch


def make_cuda_sync_fn(device: Optional[str]) -> Callable[[], None]:
    should_sync = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()

    if not should_sync:
        return lambda: None

    def _sync_fn() -> None:
        torch.cuda.synchronize()

    return _sync_fn


def _classify_generation_phase(args: Tuple[object, ...], kwargs: Dict[str, object]) -> str:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        candidate = args[0]
        if torch.is_tensor(candidate):
            input_ids = candidate
    if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
        seq_len = input_ids.shape[1]
        if seq_len <= 1:
            return "decode"
    return "prefill"


class _ForwardTimingTracker:
    def __init__(self, model: "LlavaNextForConditionalGenerationFNA", sync_fn: Callable[[], None]):
        self.model = model
        self.sync_fn = sync_fn
        self.timings = {"prefill": 0.0, "decode": 0.0}
        self._start_time: Optional[float] = None
        self._phase: str = "prefill"
        self._pre_handle = model.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._post_handle = model.register_forward_hook(self._post_hook, with_kwargs=True)

    def remove(self) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        if self._post_handle is not None:
            self._post_handle.remove()
            self._post_handle = None

    def _pre_hook(self, _module, args, kwargs):
        self.sync_fn()
        self._phase = _classify_generation_phase(args, kwargs)
        self._start_time = time.perf_counter()

    def _post_hook(self, _module, args, kwargs, _output):
        if self._start_time is None:
            return
        self.sync_fn()
        elapsed = time.perf_counter() - self._start_time
        self.timings[self._phase] += elapsed
        self._start_time = None


@contextmanager
def track_generation_timings(
    model: "LlavaNextForConditionalGenerationFNA",
    sync_fn: Callable[[], None],
):
    tracker = _ForwardTimingTracker(model, sync_fn)
    try:
        yield tracker.timings
    finally:
        tracker.remove()


def load_model_and_processor(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> Tuple[LlavaNextForConditionalGenerationFNA, LlavaNextProcessor]:
    processor_id = args.processor_id or args.model_id
    processor = LlavaNextProcessor.from_pretrained(processor_id, use_fast=False)

    fna_layers = parse_layer_selection(args.fna_layer_range, args.fna_layers, args.disable_fna)
    fna_config = {
        "fna_layers": fna_layers,
        "num_sample": args.fna_num_sample,
        "resample_every_layer": args.fna_resample_every_layer,
        "sampling_strategy": args.fna_sampling_strategy,
    }

    model_source = args.checkpoint_path or args.model_id
    logging.info("Loading LLaVA checkpoint from %s", model_source)
    model = LlavaNextForConditionalGenerationFNA.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None if args.device_map in {None, "none"} else args.device_map,
        fna_config=fna_config,
        fna_cache={},
    )
    model.eval()
    if args.device and (args.device_map in {None, "none"}):
        logging.info("Moving model to %s", args.device)
        model.to(args.device)
    return model, processor


def generate_answer(
    model: LlavaNextForConditionalGenerationFNA,
    processor: LlavaNextProcessor,
    question_payload: Dict[str, object],
    image_root: Path,
    split: str,
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> GenerationRecord:
    question_id = int(question_payload["question_id"])
    image_id = int(question_payload["image_id"])
    question_text = str(question_payload["question"])
    img_path = image_path(image_root, image_id, split)
    image = Image.open(img_path).convert("RGB")

    prompt = prepare_prompt(processor, question_text, args.system_prompt, args.answer_guidance)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = move_batch_to_device(inputs, args.device, torch_dtype)

    num_start_tokens = int(inputs["input_ids"].shape[-1])
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        })

    sync_fn = make_cuda_sync_fn(args.device)
    with torch.inference_mode():
        sync_fn()
        start = time.perf_counter()
        with track_generation_timings(model, sync_fn) as timings:
            output_ids = model.generate(**inputs, **generation_kwargs)
        sync_fn()
        latency = time.perf_counter() - start
    generation_latency = timings.get("decode", None)

    num_generated = int(output_ids.shape[-1]) - num_start_tokens
    decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = decoded.split("ASSISTANT:")[-1].strip()

    return GenerationRecord(
        question_id=question_id,
        image_id=image_id,
        question=question_text,
        answer=answer,
        raw_completion=decoded,
        latency_s=latency,
        generation_latency_s=generation_latency,
        num_starting_tokens=num_start_tokens,
        num_generated_tokens=num_generated,
    )


def read_existing_predictions(path: Path) -> List[GenerationRecord]:
    if not path.exists():
        return []
    records: List[GenerationRecord] = []
    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(GenerationRecord(**payload))
    logging.info("Resuming from %d existing predictions", len(records))
    return records


def write_record(path: Path, record: GenerationRecord) -> None:
    with path.open("a") as fp:
        fp.write(json.dumps(record.to_json()) + "\n")


def dump_submission(submission_path: Path, predictions: Sequence[GenerationRecord]) -> None:
    payload = [
        {"question_id": rec.question_id, "answer": rec.answer}
        for rec in sorted(predictions, key=lambda r: r.question_id)
    ]
    with submission_path.open("w") as fp:
        json.dump(payload, fp)


def dump_metrics(metrics_path: Path, summary: EvaluationSummary) -> None:
    with metrics_path.open("w") as fp:
        json.dump(summary.to_json(), fp, indent=2)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=level,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COCO VQA evaluation on LLaVA with Fast Nyström Attention.")
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf", help="Hugging Face model id or local path")
    parser.add_argument("--processor-id", default=None, help="Optional processor id (defaults to --model-id)")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint directory overriding --model-id")
    parser.add_argument("--questions-json", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, required=True, help="Path to COCO images root (contains split folders)")
    parser.add_argument("--image-split", default="val2014", help="COCO image split name (default: val2014)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Pass through to transformers.from_pretrained (e.g., 'auto')")
    parser.add_argument("--limit", type=int, default=None, help="Debug option to limit number of questions")
    parser.add_argument("--system-prompt", default="You are a helpful assistant for visual question answering.")
    parser.add_argument(
        "--answer-guidance",
        default="",
        #default="Answer the question using a single word or very short phrase. Respond with the answer only.",
        help="Extra instruction appended to each question to elicit short VQA-style answers (set empty to disable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fna-layer-range", default="12:32", help="Inclusive:exclusive layer range using FNA")
    parser.add_argument("--fna-layers", type=int, nargs="*", default=None, help="Explicit list of layers using FNA")
    parser.add_argument("--fna-num-sample", type=int, default=256)
    parser.add_argument("--fna-resample-every-layer", action="store_true", help="Resample landmarks before each FNA layer")
    parser.add_argument(
        "--fna-sampling-strategy",
        default="fps",
        choices=["fps", "random"],
        help="Sampling strategy used to select landmarks",
    )
    parser.add_argument("--disable-fna", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--bertscore-model-type",
        default="microsoft/deberta-large-mnli",
        help="Backbone used by the bert-score package",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bertscore-rescale", action="store_true", help="Rescale BERTScore with baseline stats")
    parser.add_argument(
        "--bertscore-device",
        default=None,
        help="Device string passed to bert-score (defaults to --device)",
    )
    parser.add_argument(
        "--bertscore-reference-strategy",
        default="majority",
        choices=["majority", "first", "concat"],
        help="How to collapse multiple VQA answers into a single BERTScore reference",
    )
    parser.add_argument(
        "--bertscore-reference-json",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file containing alternative reference answers keyed by question_id",
    )
    args = parser.parse_args()
    return args


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_coco_eval(args: argparse.Namespace) -> None:
    if not (args.questions_json and args.images_root and args.output_dir):
        raise ValueError("required arguments not specified")
    configure_logging(args.verbose)
    set_random_seed(args.seed)

    torch_dtype = dtype_from_string(args.dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    submission_path = args.output_dir / "submission.json"
    metrics_path = args.output_dir / "metrics.json"
    
    questions = load_questions(args.questions_json, args.limit)
    annotations = CocoVqaAnnotations(args.annotations_json)

    existing_predictions = read_existing_predictions(predictions_path)
    known_ids = {record.question_id for record in existing_predictions}
    predictions: List[GenerationRecord] = list(existing_predictions)

    model, processor = load_model_and_processor(args, torch_dtype)


    progress = tqdm(questions, desc="COCO VQA", unit="q")
    for payload in progress:
        question_id = int(payload["question_id"])
        if question_id in known_ids:
            continue
        record = generate_answer(model, processor, payload, args.images_root, args.image_split, args, torch_dtype)
        write_record(predictions_path, record)
        predictions.append(record)
        known_ids.add(question_id)
        maybe_empty_cuda_cache()

    dump_submission(submission_path, predictions)

    latencies = [rec.latency_s for rec in predictions]
    tokens = [rec.num_generated_tokens for rec in predictions]
    generation_latencies = [rec.generation_latency_s for rec in predictions if rec.generation_latency_s is not None]
    generation_tokens = [rec.num_generated_tokens for rec in predictions if rec.generation_latency_s is not None]
    bertscore_summary = None
    bertscore_path = args.output_dir / "bertscore.jsonl"
    bertscore_summary = compute_bertscore(predictions, annotations, bertscore_path, args)

    avg_latency = float(sum(latencies) / len(latencies)) if latencies else None
    avg_generated_tokens = float(sum(tokens) / len(tokens)) if tokens else None
    avg_tokens_per_s = None
    if (avg_latency is not None) and (avg_latency > 0) and (avg_generated_tokens is not None):
        avg_tokens_per_s = avg_generated_tokens / avg_latency
    avg_generation_latency = float(sum(generation_latencies) / len(generation_latencies)) if generation_latencies else None
    avg_generation_tokens_per_s = None
    if generation_latencies:
        total_generation_latency = float(sum(generation_latencies))
        if total_generation_latency > 0:
            avg_generation_tokens_per_s = float(sum(generation_tokens) / total_generation_latency)

    summary = EvaluationSummary(
        num_questions=len(questions),
        num_predicted=len(predictions),
        accuracy=summarize_accuracy(predictions, annotations),
        avg_latency_s=avg_latency,
        median_latency_s=float(statistics.median(latencies)) if latencies else None,
        avg_generated_tokens=avg_generated_tokens,
        avg_tokens_per_s=avg_tokens_per_s,
        avg_generation_latency_s=avg_generation_latency,
        avg_generation_tokens_per_s=avg_generation_tokens_per_s,
        bertscore_precision=(bertscore_summary or {}).get("precision"),
        bertscore_recall=(bertscore_summary or {}).get("recall"),
        bertscore_f1=(bertscore_summary or {}).get("f1"),
    )
    dump_metrics(metrics_path, summary)

    logging.info("Evaluation complete: %s", json.dumps(summary.to_json(), indent=2))




def main() -> None:
    args = parse_args()
    run_coco_eval(args)


if __name__ == "__main__":
    main()
