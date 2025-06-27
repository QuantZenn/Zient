import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from datasets import Dataset as HFDataset
from typing import List, Union, Optional, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ModelComparison.config.model_cmp import get_available_models, get_model_instance


class ModelComparison:
    def __init__(self, *model_names: str, force_download: bool = False):
        """
        Initialize ModelComparison with specified models.
        If none are provided, uses all available models.

        Args:
            *model_names: Names of models to evaluate.
            force_download (bool): Whether to force download model weights.
        """
        if not model_names:
            model_names = get_available_models()
        else:
            model_names = [name.lower() for name in model_names]

        self.models: Dict[str, object] = {}

        for name in model_names:
            if name in ["chatgpt", "zient_core"]:
                self.models[name] = get_model_instance(name)
            else:
                self.models[name] = get_model_instance(name, force_download=force_download)

    def load_dataset(self, dataset: Union[str, pd.DataFrame, HFDataset]) -> pd.DataFrame:
        if isinstance(dataset, str):
            return pd.read_csv(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return dataset.copy()
        elif isinstance(dataset, HFDataset):
            return dataset.to_pandas()
        else:
            raise TypeError("Unsupported dataset type. Use CSV path, DataFrame, or HuggingFace Dataset.")

    def evaluate(
        self,
        dataset: Union[str, Path, pd.DataFrame, HFDataset],
        text_col: str = "text",
        label_col: str = "label",
        output_path: Optional[str] = None,
        summary_path: Optional[str] = None
    ) -> pd.DataFrame:

        df = self.load_dataset(dataset)

        # Set internal default output paths if not specified
        base_path = Path(__file__).resolve().parents[3]
        base_output_dir = base_path / "outputs" 
        base_output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = base_output_dir / f"predictions.csv"
        if summary_path is None:
            summary_path = base_output_dir / f"summary.csv"

        # Convert numeric labels to string sentiment labels
        label_mapping = {
            0: "negative", 1: "neutral", 2: "positive",
            "0": "negative", "1": "neutral", "2": "positive"
        }
        df[label_col] = df[label_col].map(label_mapping)
        all_predictions = []
        summary_metrics = []

        for model_name, model in self.models.items():
            print(f"Evaluating model: {model_name}")
            start_time = time.time()

            prediction_df = model.predict_batch(df[[text_col, label_col]])
            if not isinstance(prediction_df, pd.DataFrame):
                raise ValueError("Model prediction must return a pandas DataFrame.")

            elapsed = time.time() - start_time

            pred_labels = prediction_df["predicted_label"].astype(str).tolist()
            confidence_scores = prediction_df["confidence"].tolist() if "confidence" in prediction_df.columns else [None] * len(df)
            true_labels = df[label_col].tolist()

            # Metrics
            acc = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average="weighted")
            precision = precision_score(true_labels, pred_labels, average="weighted")
            recall = recall_score(true_labels, pred_labels, average="weighted")

            summary_metrics.append({
                "model": model_name,
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "evaluation_time": round(elapsed, 4),
                "num_samples": len(df)
            })

            for i in range(len(df)):
                all_predictions.append({
                    "model": model_name,
                    "text": df[text_col].iloc[i],
                    "true_label": true_labels[i],
                    "predicted_label": pred_labels[i],
                    "confidence": confidence_scores[i],
                })

        # Save individual predictions
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(output_path, index=False)
        print(f"[INFO] Individual predictions saved to {output_path}")

        # Save summary
        summary_df = pd.DataFrame(summary_metrics)
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Model performance summary saved to {summary_path}")

        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        print(summary_df)

        return summary_df
