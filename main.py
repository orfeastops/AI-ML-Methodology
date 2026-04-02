import argparse
import pytorch_lightning as pl
from ai_ml_methodology.train import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="AI/ML Methodology for Ballistic Target Identification")
    parser.add_argument("--mode", choices=["train", "infer", "benchmark"], default="train", help="Mode to run")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with small dataset (4000 samples)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")

    args = parser.parse_args()

    pl.seed_everything(42)

    if args.wandb:
        import wandb
        wandb.init(project="ai-ml-methodology", name="ballistic-target-id")

    if args.mode == "train":
        model, results = train_and_evaluate(demo=args.demo)
        if args.wandb:
            wandb.log(results)
    elif args.mode == "infer":
        print("Inference mode not yet implemented.")
    elif args.mode == "benchmark":
        print("Benchmark mode not yet implemented.")


if __name__ == "__main__":
    main()
