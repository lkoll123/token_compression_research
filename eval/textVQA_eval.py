"""TextVQA evaluation entry-point for LLaVA + Fast Nyström Attention.

This script loads a LLaVA-NeXT checkpoint that has been instrumented with
Fast Nyström Attention (FNA) layers and runs inference on the TextVQA
validation split. It produces:

* ``predictions.jsonl`` – detailed per-question generations.
* ``submission.json`` – minimal list of ``{"question_id", "answer"}`` for
  use with the official VQA evaluation server if desired.
* ``metrics.json`` – locally computed VQA accuracy using the public metric.


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

from datasets import load_dataset, Dataset

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor

from transformers import LogitsProcessor
import torch
import string
from typing import List, Set

@dataclass
class GenerationRecord:
    question_id: str
    question: str
    answers: List[str]
    predicted_answer: str
    full_generation: str
    generation_latency_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TextVQAMetrics:
    total_questions: int
    correct_answers: int
    accuracy: float
    average_latency_s: Optional[float] = None
    median_latency_s: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def postprocess_prediction(decoded: str) -> str:
    text = decoded.strip().split("\n")[0].strip()
    text = re.sub(r"^(the answer is|answer:)\s*", "", text, flags=re.IGNORECASE)
    return normalize_text(text)

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


def write_record(path: Path, record: GenerationRecord) -> None:
    with path.open("a") as fp:
        fp.write(json.dumps(record.to_json()) + "\n")

def dump_metrics(metrics_path: Path, metrics: TextVQAMetrics) -> None:
    with metrics_path.open("w") as fp:
        json.dump(metrics.to_json(), fp, indent=2)


def maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=level,
    )

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "TextVQA Evaluation for Llava + Fast Nyström Attention (FNA)"
    )
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf", help="Hugging Face model id or local path")
    parser.add_argument("--processor-id", default=None, help="Optional processor id (defaults to --model-id)")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint directory overriding --model-id")


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
        "--textvqa-hf-dataset",
        default="lmms-lab/textvqa",
        help="HF dataset name to auto-download via datasets.load_dataset",
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)   # shorter for MCQ
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--textvqa-split", default="validation")
    parser.add_argument("--textvqa-cache-dir", type=Path, default=None)

    parser.add_argument("--textvqa-images-root", type=Path, default=None, help="Images root (local JSON mode only)")

    return parser.parse_args()

def load_textvqa_dataset(args: argparse.Namespace) -> Dataset:
    return load_dataset(
        args.textvqa_hf_dataset,
        split=args.textvqa_split,
        cache_dir=str(args.textvqa_cache_dir) if args.textvqa_cache_dir else None,
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

def prepare_prompt(
    processor: LlavaNextProcessor,
    question: str,
    has_image: bool,
    ocr_tokens: Optional[List[str]]=None,
) -> str:
    SYSTEM_PROMPT = """
    You are answering a visual question based on an image.

    Your task is to read the image carefully and provide the correct answer.

    Rules:
    - The answer must be short and concise.
    - Output only the final answer.
    - Do not include any explanation.
    - Do not include additional text or formatting.
    """

    conversation = []
    conversation.append({
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
            }
        ]
    })

    content = []
    if has_image:
        content.append({
            "type": "image",
        })

    content.append({
        "type": "text",
        "text": question,
    })

    if ocr_tokens:
        cleaned_tokens = [token.strip().lower() for token in ocr_tokens if token.strip()]
        if cleaned_tokens:
            token_text = ", ".join(cleaned_tokens)
            content.append({
                "type": "text",
                "text": f"Visible text detected in the image (OCR tokens): {token_text}",
            })

    conversation.append({
        "role": "user",
        "content": content
    })

    return processor.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
        )



def generate_answer_textvqa(
    model: LlavaNextForConditionalGenerationFNA,
    processor: LlavaNextProcessor,
    sample: Dict[str, object],
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
    sample_index: int,
) -> GenerationRecord:
    question_text = str(sample.get("question", ""))
    raw_answers = sample.get("answers", [])
    answers = []
    for ans in raw_answers:
        norm = normalize_text(str(ans))
        if norm:
            answers.append(norm)
    question_id = str(sample.get("question_id", sample_index))

    image_obj = sample.get("image", None)
    has_image = image_obj is not None
    ocr_tokens = sample.get("ocr_tokens", None)

    if has_image:
        image = image_obj.convert("RGB")
        prompt = prepare_prompt(processor, question_text, has_image=True, ocr_tokens=ocr_tokens if ocr_tokens else None)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        prompt = prepare_prompt(processor, question_text, has_image=False, ocr_tokens=ocr_tokens if ocr_tokens else None)
        inputs = processor(text=prompt, return_tensors="pt")

    inputs = move_batch_to_device(inputs, args.device, torch_dtype)

    pad_token_id = (
        processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id
    )

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

        output_ids = model.generate(**inputs, **generation_kwargs)

        sync_fn()
        latency = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return GenerationRecord(
        question_id=question_id,
        question=question_text,
        answers=answers,
        predicted_answer=postprocess_prediction(decoded),
        full_generation=decoded,
        generation_latency_s=latency,
    )

def accuracy_score(prediction: str, answers: List[str]) -> float:
    num_correct = sum([1 for ans in answers if ans == prediction])
    return min(1.0, num_correct / 3.0)

def run_textvqa_eval(args: argparse.Namespace):
    dtype = dtype_from_string(args.dtype)
    set_random_seed(args.seed)
    configure_logging(args.verbose)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "predictions.jsonl"

    model, processor = load_model_and_processor(args, dtype)
    df = load_textvqa_dataset(args)

    predictions: List[GenerationRecord] = read_existing_predictions(predictions_path)
    known_ids = set([row.question_id for row in predictions])

    progress = tqdm(enumerate(df), total=len(df), desc="Evaluating TextVQA", unit="sample")
    for i, row in progress:
        idx = str(row.get("question_id", i))
        if idx in known_ids:
            continue

        record = generate_answer_textvqa(model, processor, row, args, dtype, i)
        write_record(predictions_path, record)
        predictions.append(record)
        known_ids.add(idx)
        maybe_empty_cuda_cache()

    latencies = [rec.generation_latency_s for rec in predictions if rec.generation_latency_s is not None]
    avg_latency = float(sum(latencies) / len(latencies)) if latencies else None
    median_latency = float(statistics.median(latencies)) if latencies else None

    accuracy_scores = [accuracy_score(rec.predicted_answer, rec.answers) for rec in predictions]
    accuracy = float(sum(accuracy_scores) / len(accuracy_scores)) if accuracy_scores else 0.0
    metrics = TextVQAMetrics(
        total_questions=len(predictions),
        correct_answers=sum(1 for score in accuracy_scores if score > 0),
        accuracy=accuracy,
        average_latency_s=avg_latency,
        median_latency_s=median_latency,
    )

    dump_metrics(metrics_path, metrics)
    logging.info("Evaluation complete: %s", json.dumps(metrics.to_json(), indent=2))




def main():
    args = parse_args()
    
    run_textvqa_eval(args)



if __name__ == "__main__":
    main()