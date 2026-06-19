import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment E1")
    parser.add_argument("--results-path", type=str, default="./outputs/results")
    parser.add_argument("--models-tested", type=str, 
    choices=["all", "TiTok", "Vanilla", "FastV", "VisionZip"]
    , default="all")

def main() -> None:
    args = parse_args()
    model = args.models_tested

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
        case "FastV":
            # Run FastV model
            pass
        case "VisionZip":
            # Run VisionZip model
            pass


if __name__ == "__main__":
    main()