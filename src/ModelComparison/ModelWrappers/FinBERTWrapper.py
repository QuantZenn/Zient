import os
import torch
import pandas as pd
from typing import Union, List
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


class FinBERTWrapper:
    def __init__(self, force_download: bool = False):
        self.model_name = "finbert"
        self.model_id = "yiyanghkust/finbert-tone"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup cache and model directories
        self.base_dir = Path(__file__).resolve().parents[3]
        self.cache_dir = self.base_dir / "llm_cache" / self.model_name
        self.offload_dir = self.cache_dir / "offload"
        self.output_path = Path(__file__).resolve().parents[3] / "outputs" / f"{self.model_name}_predictions.csv"

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.offload_dir, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download or use cache
        if force_download or not any(self.cache_dir.glob("*")):
            print(f"[INFO] Downloading model: {self.model_id}")
        else:
            print(f"[INFO] Using cached model from {self.cache_dir}")

        # Load tokenizer and model with cache and force download
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            force_download=force_download
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            force_download=force_download
        )
        self.model.to(self.device)
        self.model.eval()

        # Label mapping as per FinBERT's order
        self.label_map = {0: "neutral", 1: "positive", 2: "negative"}

    def predict_batch(self, input_data: Union[pd.DataFrame, Dataset, str, List[str]]) -> pd.DataFrame:
        # Load data into DataFrame
        if isinstance(input_data, str) and input_data.endswith(".csv"):
            df = pd.read_csv(input_data)
        elif isinstance(input_data, Dataset):
            df = input_data.to_pandas()
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, list):
            df = pd.DataFrame({"text": input_data})
        else:
            raise ValueError("Unsupported input type. Must be list, CSV path, DataFrame, or Dataset.")

        if "text" not in df.columns:
            raise ValueError("Input must contain a 'text' column.")

        texts = df["text"].tolist()
        batch_size = 8
        preds, confs = [], []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                batch_preds = torch.argmax(probs, dim=1)

            for j, idx in enumerate(batch_preds):
                p = idx.item()
                preds.append(self.label_map[p])
                confs.append(round(probs[j][p].item(), 4))

        # Attach predictions
        df["predicted_label"] = preds
        df["confidence"] = confs

        # Construct output columns
        cols = ["text"]
        if "label" in df.columns:
            cols.append("label")
        cols += ["predicted_label", "confidence"]

        result_df = df[cols]
        result_df.to_csv(self.output_path, index=False)
        return result_df
