import os
import torch
import pandas as pd
from typing import List, Union
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


class ZientCoreWrapper:
    def __init__(self):
        self.model_name = "zient_core"
        self.cache_dir = Path(__file__).resolve().parents[3] / "models" / "Core"
        self.output_path = Path(__file__).resolve().parents[3] / "outputs" / f"{self.model_name}_predictions.csv"

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_path.parent, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cache_dir)
        self.model.eval()
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def _predict_single(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()
        predicted_idx = torch.argmax(probs).item()
        label = self.label_map[predicted_idx]
        confidence = round(probs[predicted_idx].item(), 4)
        return {"predicted_label": label, "confidence": confidence}

    def predict_batch(self, input_data: Union[List[str], pd.DataFrame, str, Dataset]) -> pd.DataFrame:
        if isinstance(input_data, str) and input_data.endswith(".csv"):
            input_data = pd.read_csv(input_data)
        elif isinstance(input_data, Dataset):
            input_data = input_data.to_pandas()
        elif isinstance(input_data, list):
            input_data = pd.DataFrame({"text": input_data})
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("Unsupported input type. Provide list, DataFrame, Dataset, or path to CSV.")

        if "text" not in input_data.columns:
            raise ValueError("Input must contain a 'text' column.")

        df = input_data.copy()
        predictions = []

        for i, row in df.iterrows():
            result = self._predict_single(row["text"])

            if "label" in row:
                true_label = row["label"]
                # FIXED: Properly map or retain label without crashing
                if isinstance(true_label, int):
                    result["label"] = self.label_map.get(true_label, None)
                elif isinstance(true_label, str) and true_label.lower() in {"positive", "neutral", "negative"}:
                    result["label"] = true_label.lower()
                else:
                    result["label"] = None

            result["text"] = row["text"]
            predictions.append(result)

        result_df = pd.DataFrame(predictions)
        columns = ["text", "label", "predicted_label", "confidence"] if "label" in df.columns else ["text", "predicted_label", "confidence"]
        result_df = result_df[columns]
        result_df.to_csv(self.output_path, index=False)
        return result_df
