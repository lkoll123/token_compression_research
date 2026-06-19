import argparse

from .e1a import run_e1a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment E1")
    parser.add_argument("--results-path", type=str, default="./outputs/results")
    parser.add_argument("--models-tested", type=str, 
    choices=["all", "TiTok", "Vanilla", "NuWa", "VisionZip"]
    , default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--titok-model", type=str, default="l32")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model = args.models_tested

    #e1a
    run_e1a(args)

    match model:
        case "all":
            # Run all models
            pass
        case "TiTok":
            # Run TiTok model
            pass
        case "Vanilla":
            # Run Vanilla model
            pass
        case "NuWa":
            # Run NuWa model
            pass
        case "VisionZip":
            # Run VisionZip model
            pass


if __name__ == "__main__":
    main()