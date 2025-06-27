import pandas as pd
from pathlib import Path
from typing import Union
from datasets import Dataset, load_dataset
from AutoExtractor.registry.FormatRegistry import FormatRegistry
from AutoExtractor.extractors.format_extractor import (
    adi007, fpb, kaggle_fs, std001, lilong
)

EXTRACTOR_FUNCTIONS = {
    "adi007": adi007,
    "fpb": fpb,
    "kaggle_fs": kaggle_fs,
    "std001": std001,
    "lilong": lilong,
}

DATASETS = [
    "AdiOO7/llama-2-finance",
    "AdiOO7/Llama-2",
    "nickmuchi/financial-classification",
    "lotfyhussein/finance-tweets-sentiment",
    "jppgks/twitter-financial-news-sentiment",
    "ssahir/english_finance_news",
    "lukecarlate/english_finance_news",
    "lukecarlate/general_financial_news"
]

MORE_DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "raw"

class AutoExtractor:
    def __init__(self):
        self.registry = FormatRegistry()
        self.cleaned_dir = Path(__file__).resolve().parents[3] / "dataset" / "cleaned"
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def _generate_save_path(self, keyname: str, dataset_name: str) -> Path:
        existing_files = list(self.cleaned_dir.glob(f"{keyname}_{dataset_name}_v*.csv"))
        existing_versions = []
        for f in existing_files:
            try:
                version_part = f.stem.split("_v")[-1]
                if version_part.isdigit():
                    existing_versions.append(int(version_part))
            except Exception:
                continue
        next_version = max(existing_versions, default=0) + 1
        filename = f"{keyname}_{dataset_name}_v{next_version}.csv"
        return self.cleaned_dir / filename

    def auto_extract(
        self,
        data: Union[pd.DataFrame, Dataset, str],
        sample_column: str = None,
        dataset_name: str = "unknown",
        keyname: str = None
    ) -> pd.DataFrame:

        if isinstance(data, str):
            if data.endswith(".csv"):
                df = pd.read_csv(data)
            elif data.endswith(".txt"):
                with open(data, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    raise ValueError(f"{data} is empty or malformed.")
                sample = lines[1] if len(lines) > 1 else lines[0]

                matched_format = self.registry.match(sample)
                if not matched_format:
                    raise ValueError("No matching format found for TXT sample.")

                extractor_func = EXTRACTOR_FUNCTIONS.get(matched_format["name"])
                if extractor_func is None:
                    raise ValueError(f"No extractor function registered for: {matched_format['name']}")

                cleaned_df = extractor_func(data)
                if not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
                    raise ValueError("Extractor returned empty or invalid DataFrame.")

                keyname = keyname or matched_format["name"]
                save_path = self._generate_save_path(keyname, dataset_name)
                cleaned_df.to_csv(save_path, index=False)
                return cleaned_df

            else:
                raise ValueError("Unsupported file format. Must be .csv or .txt")

        elif isinstance(data, Dataset):
            df = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Unsupported data type. Must be DataFrame, HuggingFace Dataset, or file path.")

        # On-the-fly lowercase matching
        text_candidates = [
            "text", "input", "sentence", "title", "newscontexts", 
            "headline", "summary", "content", "statement", "newscontents"
        ]

        label_candidates = [
            "label", "labels", "sentiment", "sentiment_class", "category"
        ]

        text_col = next((col for col in df.columns if col.lower() in text_candidates), None)
        label_col = next((col for col in df.columns if col.lower() in label_candidates), None)

        if not text_col:
            text_col = df.columns[0]
        if not label_col and df.shape[1] > 1:
            label_col = df.columns[1] if df.columns[1] != text_col else df.columns[-1]

        text_sample = str(df[text_col].dropna().iloc[0]) if text_col in df.columns else ""
        label_sample = str(df[label_col].dropna().iloc[0]) if label_col and label_col in df.columns else ""

        sample = f"{text_col}: {text_sample} | label: {label_sample}"

        matched_format = self.registry.match(sample)
        if not matched_format:
            raise ValueError("No matching format found for sample.")

        extractor_func = EXTRACTOR_FUNCTIONS.get(matched_format["name"])
        if extractor_func is None:
            raise ValueError(f"No extractor function registered for: {matched_format['name']}")

        cleaned_df = extractor_func(df)
        if not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
            raise ValueError("Extractor returned empty or invalid DataFrame.")

        keyname = keyname or matched_format["name"]
        save_path = self._generate_save_path(keyname, dataset_name)
        cleaned_df.to_csv(save_path, index=False)

        return cleaned_df

    def extract_all_local_datasets(self):
        for file in MORE_DATASET_PATH.glob("**/*"):
            if file.suffix.lower() in [".csv", ".txt"]:
                try:
                    dataset_name = file.stem
                    print(f"[LOCAL] Extracting: {file.name}")
                    self.auto_extract(str(file), dataset_name=dataset_name)
                except Exception as e:
                    print(f"[WARN] Skipped {file.name}: {e}")

    def extract_all_hf_datasets(self):
        for dataset_id in DATASETS:
            try:
                print(f"[HF] Extracting {dataset_id}")
                ds = load_dataset(dataset_id, split="train")
                dataset_name = dataset_id.split("/")[-1]
                self.auto_extract(ds, dataset_name=dataset_name)
            except Exception as e:
                print(f"[WARN] Failed on {dataset_id}: {e}")

    def extract_all(self):
        self.extract_all_local_datasets()
        self.extract_all_hf_datasets()
