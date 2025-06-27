import argparse
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional
from ModelComparison.Model.ModelComparison import ModelComparison


def internal_evaluate(dataset_path: str, models: Optional[List[str]] = None, force_download: bool = False) -> pd.DataFrame:
    """
    Internal programmatic usage of model evaluation.
    Evaluates all available models if none are specified.
    Returns result DataFrame.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    comparator = ModelComparison(*(models or []), force_download=force_download)
    results_df = comparator.evaluate(dataset=dataset_path)

    return results_df


def run_comparison():
    parser = argparse.ArgumentParser(description="Run sentiment model comparison")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to evaluate (e.g. chatgpt deepseek zientcore). Default: all available.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset CSV or HuggingFace Dataset name.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save the comparison results as CSV.")
    parser.add_argument("--force_download", action="store_true",
                        help="Force re-download of model files.")

    args = parser.parse_args()

    print("=== Zient Model Comparison Executor ===")
    print(f"Models: {args.models if args.models else 'ALL'}")
    print(f"Dataset: {args.dataset}")
    print(f"Force download: {args.force_download}")
    print("----------------------------------------")

    comparator = ModelComparison(*args.models if args.models else [],
                                  force_download=args.force_download)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join("dataset", "model_comparison", f"{timestamp}.csv")
    output_path = args.output or default_output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results_df = comparator.evaluate(dataset=args.dataset, output_path=output_path)

    print("\n=== Evaluation Report Results ===")
    print(results_df.head(10).to_string(index=False))
    print(f"\nFull results saved to: {output_path}")



