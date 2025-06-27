import pandas as pd
from datasets import Dataset


def adi007(dataset_or_path):
    if isinstance(dataset_or_path, str):
        if dataset_or_path.endswith(".csv"):
            df = pd.read_csv(dataset_or_path)
        elif dataset_or_path.endswith(".txt"):
            with open(dataset_or_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            df = pd.DataFrame({"text": lines})
        else:
            raise ValueError("Unsupported file format. Must be .csv or .txt")
    elif isinstance(dataset_or_path, Dataset):
        df = dataset_or_path.to_pandas()
    elif isinstance(dataset_or_path, pd.DataFrame):
        df = dataset_or_path.copy()
    else:
        raise TypeError("Input must be a Hugging Face Dataset, DataFrame, or a .csv/.txt file path.")

    # Extract
    cleaned_rows = []
    for row in df.itertuples():
        full_text = getattr(row, "text", "")
        if "### Human:" in full_text:
            extracted = full_text.split("### Human:")[-1].strip()
            label = None
            if "### Assistant:" in extracted:
                parts = extracted.split("### Assistant:")
                extracted = parts[0].strip()
                label = parts[1].strip().lower()
                cleaned_rows.append({"text": extracted, "label": label})
            else:
                cleaned_rows.append({"text": extracted})

    return pd.DataFrame(cleaned_rows)

def fpb(data: pd.DataFrame | str | Dataset) -> pd.DataFrame:
    if isinstance(data, Dataset):
        data = data.to_pandas()
    elif isinstance(data, str):
        with open(data, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif isinstance(data, pd.DataFrame):
        lines = data.iloc[:, 0].dropna().astype(str).tolist()
    else:
        raise TypeError("Input must be a CSV/TXT path, pandas DataFrame, or HF Dataset")

    label_map = {"@negative": "negative", "@neutral": "neutral", "@positive": "positive"}
    rows = []

    for line in lines:
        line = line.strip()
        if not line or "@" not in line:
            continue
        try:
            text, raw_label = line.rsplit("@", 1)
            label = "@" + raw_label.strip().lower()
            if label in label_map:
                rows.append({
                    "text": text.strip().strip('"'),
                    "label": label_map[label]
                })
        except ValueError:
            continue

    return pd.DataFrame(rows)

def kaggle_fs(data: pd.DataFrame | str | Dataset) -> pd.DataFrame:
    if isinstance(data, str):
        if data.endswith(".csv"):
            data = pd.read_csv(data)
        elif data.endswith(".txt"):
            with open(data, "r", encoding="utf-8", errors="replace") as f:
                lines = [line.strip() for line in f if line.strip()]
            rows = []
            for line in lines:
                if "," not in line:
                    continue
                try:
                    text_part, label_part = line.rsplit(",", 1)
                    text = text_part.strip().strip('"')
                    label = label_part.strip().lower()
                    if label in {"positive", "neutral", "negative"}:
                        rows.append({"text": text, "label": label})
                except Exception:
                    continue
            return pd.DataFrame(rows)
        else:
            raise ValueError("Unsupported file format for kaggle_fs. Must be .csv or .txt.")

    elif isinstance(data, Dataset):
        data = data.to_pandas()
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be CSV path, .txt path, pandas DataFrame, or HF Dataset")

    # Standard Kaggle format
    if "Sentence" not in data.columns or "Sentiment" not in data.columns:
        raise ValueError("Input must contain 'Sentence' and 'Sentiment' columns.")

    df = data[["Sentence", "Sentiment"]].copy()
    df.rename(columns={"Sentence": "text", "Sentiment": "label"}, inplace=True)
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["positive", "neutral", "negative"])]

    return df

def lilong(data: pd.DataFrame | str | Dataset) -> pd.DataFrame:
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif isinstance(data, Dataset):
        data = data.to_pandas()
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be CSV path, pandas DataFrame, or HF Dataset")

    required_cols = {"input", "output"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Input must contain columns: {required_cols}")

    df = data[["input", "output"]].copy()
    df.rename(columns={"input": "text", "output": "label"}, inplace=True)

    label_map = {
        "neuter": "neutral",
        "front": "positive",
        "negative": "negative"
    }
    df["label"] = df["label"].str.lower().map(label_map)
    df = df.dropna(subset=["label"])

    return df

def std001(data: pd.DataFrame | str | Dataset) -> pd.DataFrame:
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif isinstance(data, Dataset):
        data = data.to_pandas()

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a CSV path, pandas DataFrame, or Hugging Face Dataset")

    # Fully extended column name matching
    text_candidates = [
        "text", "input", "sentence", "title", "newscontexts",
        "headline", "summary", "content", "statement", "newscontents"
    ]
    label_candidates = [
        "label", "labels", "sentiment", "sentiment_class", "category", "output"
    ]

    # Convert all column names to lowercase for flexible matching
    columns_lower = {col.lower(): col for col in data.columns}

    text_col = next((columns_lower.get(c) for c in text_candidates if c in columns_lower), None)
    label_col = next((columns_lower.get(c) for c in label_candidates if c in columns_lower), None)

    if text_col is None or label_col is None:
        raise ValueError("Data must contain a valid text and label column.")

    df = data[[text_col, label_col]].copy()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)

    label_map = {
        "0": "negative", 0: "negative", "negative": "negative",
        "1": "neutral",  1: "neutral",  "neutral": "neutral",
        "2": "positive", 2: "positive", "positive": "positive",
        "pos": "positive", "neg": "negative", "neu": "neutral"
    }

    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df.dropna(subset=["label"])

    return df


