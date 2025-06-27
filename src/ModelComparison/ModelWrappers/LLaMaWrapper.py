import os
import torch
import pandas as pd
from typing import List, Union
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLaMAaWrapper:
    def __init__(self, force_download: bool = False):
        self.model_name = "llama"
        self.model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.cache_dir = Path(__file__).resolve().parents[2] / "llm_cache" / self.model_name
        self.offload_dir = self.cache_dir / "offload"
        self.output_path = Path(__file__).resolve().parents[3] / "outputs" / f"{self.model_name}_predictions.csv"

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.offload_dir, exist_ok=True)
        os.makedirs(self.output_path.parent, exist_ok=True)

        if force_download or not any(self.cache_dir.iterdir()):
            print(f"Downloading model: {self.model_id}")
        else:
            print(f"Using cached model from {self.cache_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            offload_folder=str(self.offload_dir)
        )
        self.model.eval()

    def _parse_response(self, response_text: str) -> Union[str, None]:
        text = response_text.lower().strip()
        if "positive" in text:
            return "positive"
        elif "negative" in text:
            return "negative"
        elif "neutral" in text:
            return "neutral"
        return None  # Skip unrecognized

    def _extract_qualitative_confidence(self, text: str) -> str:
        text = text.lower().strip()
        if any(x in text for x in ["most", "extremely"]):
            return "most"
        elif "very" in text:
            return "very"
        elif "somewhat" in text:
            return "somewhat"
        return "none"

    def _predict_single(self, text: str) -> Union[dict, None]:
        prompt = (
            "You are a financial sentiment analysis model. "
            "Classify the sentiment in the following financial text as Positive, Neutral, or Negative. "
            "Respond with only the sentiment word.\n\n"
            f"Text: {text}\nSentiment:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        label = self._parse_response(decoded)
        if label is None:
            return None

        # Follow-up prompt for qualitative confidence
        followup_prompt = (
            f"How confident are you in your classification of the sentiment above? Choose one of: none, somewhat, very, most.\n"
            f"Sentiment: {label}"
        )
        followup_inputs = self.tokenizer(followup_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            followup_outputs = self.model.generate(**followup_inputs, max_new_tokens=10)
        followup_decoded = self.tokenizer.decode(followup_outputs[0], skip_special_tokens=True)

        return {
            "predicted_label": label,
            "confidence": self._extract_qualitative_confidence(followup_decoded)
        }

    def predict_batch(self, input_data: Union[List[str], pd.DataFrame, str, Dataset]) -> pd.DataFrame:
        if isinstance(input_data, str) and input_data.endswith(".csv"):
            input_data = pd.read_csv(input_data)
        elif isinstance(input_data, Dataset):
            input_data = input_data.to_pandas()
        elif isinstance(input_data, list):
            input_data = pd.DataFrame({"text": input_data})
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("Unsupported input type. Must be list, DataFrame, Dataset, or CSV path.")

        if "text" not in input_data.columns:
            raise ValueError("Input must contain a 'text' column.")

        df = input_data.copy()
        results = []

        for i, row in df.iterrows():
            result = self._predict_single(row["text"])
            if result is None:
                continue  # Skip invalid/unclassifiable
            entry = {
                "text": row["text"],
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"]
            }
            if "label" in row and pd.notna(row["label"]):
                entry["label"] = row["label"]
            results.append(entry)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            if "label" in result_df.columns:
                result_df = result_df[["text", "label", "predicted_label", "confidence"]]
            else:
                result_df = result_df[["text", "predicted_label", "confidence"]]
            result_df.to_csv(self.output_path, index=False)
        return result_df
