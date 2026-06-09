""" VQA evaluation entrypoint for Llava + Fast Nyström Attention (FNA).

Evaluated on following benchmarks:
 - VQAv2
 - TextVQA
 - QK-VQA
 - ScienceQA

 This script is a multimodal endpoint that parses arguments, and runs evaluation using 
 methods imported from respective benchmark evaluation modules.

 """

from __future__ import annotations
import argparse 
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coco_vqa_eval import run_coco_eval
from scienceQA_eval import run_scienceqa_eval
from textVQA_eval import run_textvqa_eval





# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQA Evaluation for Llava + Fast Nyström Attention (FNA)"
    )   
    parser.add_argument(
        "--benchmark", 
        required=True,
        choices=["vqav2", "textvqa", "qk-vqa", "scienceqa"],
        help="Which Benchmark to run"
        )

    # ---------------------------------------------------------------------------
    # General + Coco Vqa Eval Args
    # ---------------------------------------------------------------------------
    parser.add_argument("--model-id", default="llava-hf/llava-v1.6-vicuna-7b-hf", help="Hugging Face model id or local path")
    parser.add_argument("--processor-id", default=None, help="Optional processor id (defaults to --model-id)")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local checkpoint directory overriding --model-id")
    parser.add_argument("--questions-json", type=Path, default=None)
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None, help="Path to COCO images root (contains split folders)")
    parser.add_argument("--image-split", default="val2014", help="COCO image split name (default: val2014)")
    parser.add_argument("--output-dir", type=Path, default=None)
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

    # ---------------------------------------------------------------------------
    # Additional ScienceQA Eval Args
    # ---------------------------------------------------------------------------

    scqa = parser.add_argument_group("ScienceQA Evaluation Arguments")
    scqa.add_argument("--scienceqa-split", default="validation")
    scqa.add_argument("--scienceqa-cache-dir", type=Path, default=None)
    scqa.add_argument("--scienceqa-images-root", type=Path, default=None, help="Images root (local JSON mode only)")
    scqa.add_argument(
        "--scienceqa-hf-dataset",
        default="lmms-lab/ScienceQA-IMG",
        help="HF dataset name to auto-download via datasets.load_dataset",
    )

    txtvqa = parser.add_argument_group("TextVQA Evaluation Arguments")
    txtvqa.add_argument("--textvqa-split", default="validation")
    txtvqa.add_argument("--textvqa-cache-dir", type=Path, default=None)
    txtvqa.add_argument("--textvqa-images-root", type=Path, default=None, help="Images root (local JSON mode only)")
    txtvqa.add_argument(
         "--textvqa-hf-dataset",
        default="lmms-lab/textvqa",
        help="HF dataset name to auto-download via datasets.load_dataset",
    )

    return parser.parse_args()





    

# ---------------------------------------------------------------------------
# Orchestration - main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    benchmark = args.benchmark.lower()

    if benchmark == "vqav2":
        run_coco_eval(args)

    elif benchmark == "textvqa":
        run_textvqa_eval(args)

    elif benchmark == "qk-vqa":
        raise NotImplementedError("QK-VQA evaluation not yet implemented.")
    elif benchmark == "scienceqa":
        run_scienceqa_eval(args)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


if __name__ == "__main__":
    main()