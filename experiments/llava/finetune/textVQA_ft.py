"""
This script loads a Llava-Next checkpoint and fine-tunes it on the TextVQA dataset.
"""

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from functools import partial
import argparse
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
from collections import Counter
from peft import LoraConfig, get_peft_model

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_nystrom_attention import LlavaNextForConditionalGenerationFNA
from transformers import LlavaNextProcessor


SYSTEM_PROMPT = """
You are answering a visual question based on an image.

Your task is to read the image carefully and provide the correct answer.

Rules:
- The answer must be short and concise.
- Output only the final answer.
- Do not include any explanation.
- Do not include additional text or formatting.
"""

class DataRow():
    def __init__(self, data: Dict, processor: LlavaNextProcessor):
        self.image = self.__extract_image(data)
        self.prompt = self.__extract_prompt(data, processor)
        self.gt = self.__extract_gt(data)


    def __extract_image(self, data: Dict) -> Image.Image:
        image = data.get("image", None)
        return image.convert("RGB") if image else None
            

    def __extract_gt(self, data: Dict) -> str:
        answers = data.get("answers", None)
        
        if answers is None or len(answers) == 0:
            raise ValueError("Data row is missing 'answers' field or it is empty")
        answers = [ans.strip() for ans in answers if isinstance(ans, str) and ans.strip()]
        counts = Counter(answers)
        maxCount = -1
        currAns = None
        for ans, count in counts.items():
            if count > maxCount:
                maxCount = count
                currAns = ans
        return currAns
        
    def __extract_prompt(self, data: Dict, processor: LlavaNextProcessor) -> str:
        question = data.get("question", None)

        if question is None:
            raise ValueError("Data row is missing 'question' field")


        conversation = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                    {"type": "text", "text": "Answer:"},
                ],
            },
        ]

        return processor.apply_chat_template(conversation, add_generation_prompt=False)

def print_lora_candidates(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            print(name, module)

def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False

def apply_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_textvqa_dataset(args: argparse.Namespace, split: str) -> Dataset:
    return load_dataset(
        args.textvqa_hf_dataset,
        split=split,
        cache_dir=str(args.textvqa_cache_dir) if args.textvqa_cache_dir else None,
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def collate_fn(batch: List[DataRow], args: argparse.Namespace, processor: LlavaNextProcessor) -> Dict[str, torch.Tensor]:
    batch = [r for r in batch if r.prompt is not None and r.gt is not None]
    if len(batch) == 0:
        raise ValueError("All rows in the batch are invalid (missing prompt or ground truth)")
    prompts = [row.prompt for row in batch]
    images = [row.image for row in batch]

    TARGET = 336
    images = [
        img.resize((TARGET, TARGET)) if img is not None
        else Image.new("RGB", (TARGET, TARGET), (0, 0, 0))
        for img in images
    ]
    ground_truths = [row.gt for row in batch]

    full_prompts = [f"{prompt} {ground_truth}" for prompt, ground_truth in zip(prompts, ground_truths)]

    inputs = processor(
        text=full_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    labels = inputs["input_ids"].clone()
    tok = processor.tokenizer

    for i, prompt in enumerate(prompts):
        prompt_ids = tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,  
            truncation=False,
        )["input_ids"][0]

        prompt_len = len(prompt_ids)
        labels[i, :prompt_len] = -100

    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels

    return inputs


def save_hf_checkpoint(
    model: LlavaNextForConditionalGenerationFNA, 
    processor: LlavaNextProcessor, 
    output_dir: Path, 
    step: int) -> None:
    out_dir = output_dir / f"checkpoint_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model

    unwrapped.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    logging.info("Saved checkpoint %d to %s", step, str(out_dir))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llava-Next + FNA on TextVQA")
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf", help="Hugging Face model id or local path")
    parser.add_argument("--processor-id", default=None, help="Optional processor id (defaults to --model-id)")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint directory overriding --model-id")
    parser.add_argument("--output-dir", type=Path, required=True)

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

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--grad-checkpointing", action="store_true")

    parser.add_argument("--textvqa-hf-dataset", default="lmms-lab/textvqa", help="Hugging Face dataset id for TextVQA")
    parser.add_argument("--textvqa-cache-dir", type=Path, default=None)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)


    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    set_seed(args.seed)

    dtype = dtype_from_string(args.dtype)
    model, processor = load_model_and_processor(args, dtype)
    if args.use_lora:
        #print_lora_candidates(model)
        if hasattr(model, "vision_tower"):
            freeze_module(model.vision_tower)
        if hasattr(model, "multi_modal_projector"):
            freeze_module(model.multi_modal_projector)
        if hasattr(model, "lm_head"):
            freeze_module(model.lm_head)
        model = apply_lora(model, args)

    train_ds = load_textvqa_dataset(args, split="train")
    test_ds = load_textvqa_dataset(args, split="validation")

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    if args.max_eval_samples is not None:
        test_ds = test_ds.select(range(min(args.max_eval_samples, len(test_ds))))

    train_rows = [DataRow(data, processor) for data in train_ds]
    test_rows = [DataRow(data, processor) for data in test_ds]

    collate = partial(collate_fn, args=args, processor=processor)

    train_loader = DataLoader(
        train_rows, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate
        )
    test_loader = DataLoader(
        test_rows, 
        batch_size=args.batch_size, 
        collate_fn=collate)

    model.train()
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
    device = torch.device(args.device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optim = AdamW(trainable_params, lr=args.lr)
    optim.zero_grad(set_to_none=True)

    global_step = 0

    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        logging.info("Starting epoch %d/%d", epoch + 1, args.num_epochs)
        batch_bar = tqdm(enumerate(train_loader), desc="Batches", leave=False)
        i = -1
        for i, batch in batch_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss/args.grad_accum_steps
            loss.backward()

            if (i+1) % args.grad_accum_steps == 0:
                global_step +=1
                optim.step()
                optim.zero_grad(set_to_none=True)
                if global_step % args.save_every == 0:
                    save_hf_checkpoint(model, processor, args.output_dir, global_step)

            batch_bar.set_postfix(loss=f"{(loss.item() * args.grad_accum_steps):.4f}")

        remainder = (i+1) % args.grad_accum_steps
        if remainder != 0:
            global_step +=1
            optim.step()
            optim.zero_grad(set_to_none=True)
            if global_step % args.save_every == 0:
                save_hf_checkpoint(model, processor, args.output_dir, global_step)
    save_hf_checkpoint(model, processor, args.output_dir, global_step)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    batch_bar = tqdm(enumerate(test_loader), desc="Evaluating", leave=False)
    for i, batch in batch_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        total_loss += outputs.loss.item()
        num_batches += 1


    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    logging.info("Evaluation completed. Average loss: %.4f", avg_loss)





if __name__ == "__main__":
    main()