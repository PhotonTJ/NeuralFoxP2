"""
Neural FOXP2 - Main Pipeline Runner

Complete pipeline for language neuron localization, steering direction
identification, and intervention edit rule optimization.

Based on the Neural FOXP2 paper methodology.
"""
import argparse
import pickle
import os
from pathlib import Path

from config import config
from models import load_model_and_tokenizer
from data import load_matched_prompts, build_language_token_sets
from stage1 import run_stage1
from stage2 import run_stage2
from stage3 import run_stage3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural FOXP2: Language Steering via SAE Interventions"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="HuggingFace API token for model access"
    )
    
    parser.add_argument(
        "--layers",
        type=str,
        default="8-23",
        help="Layer range to analyze (e.g., '8-23' or '18')"
    )
    
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=2500,
        help="Number of parallel prompts to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="SAE training epochs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from saved checkpoint"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "1", "2", "3"],
        default="all",
        help="Which stage to run"
    )
    
    return parser.parse_args()


def parse_layer_range(layer_str: str):
    """Parse layer range string like '8-23' or '18' into list."""
    if "-" in layer_str:
        start, end = map(int, layer_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(layer_str)]


def save_checkpoint(data: dict, output_dir: str, name: str):
    """Save checkpoint to pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: str):
    """Load checkpoint from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    
    # Update config from args
    if args.hf_token:
        config.hf_token = args.hf_token
    config.layer_range = parse_layer_range(args.layers)
    config.n_prompts = args.n_prompts
    config.epochs = args.epochs
    
    print("="*70)
    print("Neural FOXP2 Pipeline")
    print("="*70)
    print(f"Layer range: {config.layer_range}")
    print(f"N prompts: {config.n_prompts}")
    print(f"SAE epochs: {config.epochs}")
    print("="*70)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load data
    matched_prompts = load_matched_prompts()
    V_hi, V_en = build_language_token_sets(tokenizer)
    
    # Resume or run pipeline
    stage1_results = None
    stage2_results = None
    stage3_results = None
    
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        checkpoint = load_checkpoint(args.resume_from)
        stage1_results = checkpoint.get("stage1")
        stage2_results = checkpoint.get("stage2")
        stage3_results = checkpoint.get("stage3")
    
    # Stage I: Localize Language Neurons
    if args.stage in ["all", "1"] and stage1_results is None:
        stage1_results = run_stage1(
            model, tokenizer, matched_prompts,
            V_hi, V_en, config.layer_range
        )
        save_checkpoint({"stage1": stage1_results}, args.output_dir, "stage1_checkpoint")
    
    # Stage II: Identify Steering Directions
    if args.stage in ["all", "2"] and stage2_results is None:
        if stage1_results is None:
            raise ValueError("Stage I results required. Run Stage I first or resume from checkpoint.")
        
        stage2_results = run_stage2(
            model, tokenizer, matched_prompts,
            stage1_results["sae_per_layer"],
            stage1_results["language_neurons_per_layer"],
            config.layer_range
        )
        save_checkpoint({
            "stage1": stage1_results,
            "stage2": stage2_results
        }, args.output_dir, "stage2_checkpoint")
    
    # Stage III: Intervention Edit Rule
    if args.stage in ["all", "3"] and stage3_results is None:
        if stage1_results is None or stage2_results is None:
            raise ValueError("Stage I and II results required.")
        
        stage3_results = run_stage3(
            model, tokenizer, matched_prompts,
            stage1_results["sae_per_layer"],
            stage1_results["language_neurons_per_layer"],
            stage2_results["layer_results"],
            stage2_results["intervention_window"],
            V_hi, V_en
        )
        save_checkpoint({
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": stage3_results
        }, args.output_dir, "final_checkpoint")
    
    # Print final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    if stage1_results:
        print(f"\nStage I: Trained {len(stage1_results['sae_per_layer'])} SAEs")
        for layer in stage1_results['language_neurons_per_layer']:
            n_neurons = len(stage1_results['language_neurons_per_layer'][layer])
            print(f"  Layer {layer}: {n_neurons} language neurons")
    
    if stage2_results:
        print(f"\nStage II: Intervention window = {stage2_results['intervention_window']}")
        print(f"  Window score: {stage2_results['window_score']:.4f}")
    
    if stage3_results:
        print(f"\nStage III: Best params = λ={stage3_results['best_params'][0]:.2f}, "
              f"β={stage3_results['best_params'][1]:.2f}")
        print(f"  Defaultness gain: {stage3_results['best_gain']:.4f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
