import os
import torch
import pandas as pd
from typing import List, Union
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class CommandRPlusWrapper:
    def __init__(self, force_download: bool = False):
        self.model_name = "commandrplus"
        self.model_id = "CohereForAI/c4ai-command-r-plus"
        self.cache_dir = Path(__file__).resolve().parents[3] / "llm_cache" / self.model_name
        self.offload_dir = self.cache_dir / "offload"
        self.output_path = f"outputs/{self.model_name}_predictions.csv"

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.offload_dir, exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        if force_download or not any(self.cache_dir.iterdir()):
            print(f"[INFO] Downloading model: {self.model_id}")
        else:
            print(f"[INFO] Using cached model from {self.cache_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            offload_folder=str(self.offload_dir)
        )
        self.model.eval()

    def _parse_response(self, response_text: str) -> Union[dict, None]:
        response_text = response_text.lower().strip()
        if "positive" in response_text:
            return {"predicted_label": "positive", "confidence": None}
        elif "negative" in response_text:
            return {"predicted_label": "negative", "confidence": None}
        elif "neutral" in response_text:
            return {"predicted_label": "neutral", "confidence": None}
        return None  # ✅ Skip if response is invalid

    def _extract_qualitative_confidence(self, text: str) -> str:
        text = text.lower()
        if "most" in text or "extremely" in text:
            return "most"
        elif "very" in text:
            return "very"
        elif "somewhat" in text:
            return "somewhat"
        return "none"

    def _predict_single(self, text: str) -> Union[dict, None]:
        sentiment_prompt = (
            "You are a financial sentiment analysis model. "
            "Classify the sentiment in the following financial text as Positive, Neutral, or Negative. "
            "Respond with only the sentiment word."
            f"\n\nText: {text}\nSentiment:"
        )
        inputs = self.tokenizer(sentiment_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=10)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        result = self._parse_response(decoded)
        if result is None:
            return None  # ❌ Skip if invalid

        # Estimate qualitative confidence
        confidence_prompt = (
            f"How confident are you in your classification of the sentiment above? "
            f"Choose one of: none, somewhat, very, most.\nSentiment: {result['predicted_label']}"
        )
        conf_inputs = self.tokenizer(confidence_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            conf_output = self.model.generate(**conf_inputs, max_new_tokens=10)
        conf_decoded = self.tokenizer.decode(conf_output[0], skip_special_tokens=True)
        result["confidence"] = self._extract_qualitative_confidence(conf_decoded)

        return result

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
            raise ValueError("DataFrame must contain a 'text' column.")

        df = input_data.copy()
        valid_preds = []
        valid_indices = []

        for i, text in enumerate(df["text"].tolist()):
            result = self._predict_single(text)
            if result:
                valid_preds.append(result)
                valid_indices.append(i)

        df = df.iloc[valid_indices].copy()
        df["predicted_label"] = [r["predicted_label"] for r in valid_preds]
        df["confidence"] = [r["confidence"] for r in valid_preds]

        if "label" not in df.columns:
            df["label"] = None

        result_df = df[["text", "label", "predicted_label", "confidence"]]
        result_df.to_csv(self.output_path, index=False)
        return result_df
