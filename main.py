# Install dependencies via requirements file (notebooks use !pip usage).
# pip install -r REQUIREMENTS.txt

import argparse
import os
import pytorch_lightning as pl
from ai_ml_methodology.train import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="AI/ML Methodology for Ballistic Target Identification")
    parser.add_argument("--mode", choices=["train", "infer", "benchmark"], default="train", help="Mode to run")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with small dataset")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--data-source", choices=["missile", "aircraft"], default="missile", 
                       help="Data source: 'missile' (simulated) or 'aircraft' (real OpenSky-like)")

    args = parser.parse_args()

    pl.seed_everything(42)

    if args.wandb:
        import wandb
        wandb.init(project="ai-ml-methodology", name=f"ballistic-target-id-{args.data_source}")

    if args.mode == "train":
        model, results = train_and_evaluate(demo=args.demo, data_source=args.data_source)
        if args.wandb:
            wandb.log({"data_source": args.data_source, **results})
    elif args.mode == "infer":
        # Placeholder for inference mode
        print("Inference mode not yet implemented.")
    elif args.mode == "benchmark":
        # Placeholder for benchmark mode
        print("Benchmark mode not yet implemented.")


if __name__ == "__main__":
    main()
