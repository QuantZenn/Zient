import os
import time
import openai
import pandas as pd
from pathlib import Path
from datasets import Dataset
from typing import List, Union
from ModelComparison.config.evn import OPENAI_API_KEY


class ChatGPTWrapper:
    def __init__(self, model: str = "gpt-4o"):
        openai.api_key = OPENAI_API_KEY
        self.model = model
        self.system_prompt = (
            "You are a financial sentiment analysis model. "
            "Given a financial news article, tweet, or report, classify the overall sentiment "
            "towards the market or stock mentioned as Positive, Neutral, or Negative."
        )
        self.output_path = Path(__file__).resolve().parents[3] / "outputs" / "chatgpt_predictions.csv"
        os.makedirs("outputs", exist_ok=True)

    def _parse_label_response(self, response_text: str) -> str:
        response_text = response_text.lower().strip()
        if "positive" in response_text:
            return "positive"
        elif "negative" in response_text:
            return "negative"
        elif "neutral" in response_text:
            return "neutral"
        else:
            return None

    def _parse_confidence_response(self, response_text: str) -> str:
        response_text = response_text.lower()
        if "most" in response_text or "extremely" in response_text:
            return "most"
        elif "very" in response_text:
            return "very"
        elif "somewhat" in response_text:
            return "somewhat"
        elif "none" in response_text:
            return "none"
        return "none"

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

        for text in df["text"]:
            try:
                # First prompt: sentiment classification
                label_resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.2
                )
                label = self._parse_label_response(label_resp["choices"][0]["message"]["content"])

                # Second prompt: confidence assessment
                confidence_prompt = (
                    "On a scale of none, somewhat, very, or most, how confident are you in your previous answer?"
                )
                conf_resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial analysis assistant."},
                        {"role": "user", "content": confidence_prompt}
                    ],
                    temperature=0.0
                )
                confidence = self._parse_confidence_response(conf_resp["choices"][0]["message"]["content"])

            except Exception as e:
                print(f"[ERROR] Failed to process input: {text[:60]}...\n{e}")
                label = None
                confidence = None

            predictions.append({
                "text": text,
                "label": None,  # placeholder, overwritten below if label exists in input
                "predicted_label": label,
                "confidence": confidence
            })
            time.sleep(1.2)

        result_df = pd.DataFrame(predictions)

        if "label" in df.columns:
            result_df["label"] = df["label"]

        # Ensure standard column order
        result_df = result_df[["text", "label", "predicted_label", "confidence"]]

        result_df.to_csv(self.output_path, index=False)
        return result_df
