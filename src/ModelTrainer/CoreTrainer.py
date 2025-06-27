import os
import json
import shutil
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, ClassLabel, Features, Value
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

class CoreTrainer:
    def __init__(self, model_name="roberta-base", num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainer = None

        self.base_path = Path(__file__).resolve().parents[0]
        self.input_path = self.base_path / "dataset" / "cleaned"
        self.output_path = self.base_path / "dataset" / "training" / "Core" / "cleaned.csv"
        self.model_output_path = self.base_path / "models" / "Core"
        self.log_path = self.base_path / "logs" / "models" / "Core"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def clean_and_save_dataset(self):
        all_files = list(self.input_path.glob("*.csv"))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_path}")

        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)

        df["label"] = df["label"].str.lower().str.strip().str.replace(".", "", regex=False)
        df = df.drop_duplicates(subset=["text", "label"])

        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df = df[df["label"].isin(label_map)]
        df["label"] = df["label"].map(label_map)

        df = self._rebalance_dataset(df)

        df.to_csv(self.output_path, index=False)
        print(f"[INFO] Cleaned dataset saved to: {self.output_path}")

    def _rebalance_dataset(self, df):
        majority_class = df["label"].value_counts().idxmax()
        majority_df = df[df["label"] == majority_class]
        minority_dfs = [df[df["label"] == c] for c in df["label"].unique() if c != majority_class]
        upsampled = [resample(d, replace=True, n_samples=len(majority_df), random_state=42) for d in minority_dfs]
        return pd.concat([majority_df] + upsampled).sample(frac=1, random_state=42).reset_index(drop=True)

    def load_and_tokenize_dataset(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        df = pd.read_csv(self.output_path)
        if df["label"].isnull().any():
            raise ValueError("Some labels are missing after CSV load.")

        features = Features({
            "text": Value("string"),
            "label": ClassLabel(names=["negative", "neutral", "positive"])
        })

        dataset = Dataset.from_pandas(df[["text", "label"]], features=features)
        dataset = dataset.add_column("orig_index", list(range(len(dataset))))

        tokenized = dataset.map(
            lambda x: self.tokenizer(
                [str(t) if t is not None else "" for t in x["text"]],
                truncation=True,
                padding="max_length",
                max_length=128
            ),
            batched=True
        )
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        split = tokenized.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")

        train_indices = split["train"]["orig_index"]
        test_indices = split["test"]["orig_index"]

        raw_df = pd.read_csv(self.output_path)
        raw_train_df = raw_df.iloc[train_indices].reset_index(drop=True)
        raw_test_df = raw_df.iloc[test_indices].reset_index(drop=True)

        raw_train_df.to_csv(self.output_path.parent / "train.csv", index=False)
        raw_test_df.to_csv(self.output_path.parent / "test.csv", index=False)

        self._save_token_metadata(dataset, self.tokenizer, split=split)

        return split

    def _save_token_metadata(self, dataset, tokenizer, max_length=128, split=None):
        texts = dataset["text"]
        labels = dataset["label"]
        encodings = tokenizer(list(map(str, texts)), truncation=True, padding=False, return_length=True)
        lengths = encodings["length"]
        total_tokens = sum(lengths)
        avg_tokens = total_tokens / len(texts)

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        class_counts = defaultdict(int)
        token_counts = defaultdict(int)

        for l, t in zip(labels, lengths):
            label = label_map[l]
            class_counts[label] += 1
            token_counts[label] += t

        meta = {
            "tokenizer": self.model_name,
            "max_length": max_length,
            "total_samples": len(texts),
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens,
            "class_distribution": dict(class_counts),
            "token_histogram_per_class": dict(token_counts),
            "train_size": len(split["train"]) if split else None,
            "test_size": len(split["test"]) if split else None,
            "timestamp": datetime.now().isoformat()
        }

        meta_path = self.base_path / "meta" / "Core" / "tokens.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Token metadata saved to: {meta_path}")

    @staticmethod
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        report = classification_report(p.label_ids, preds, output_dict=True)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": report["weighted avg"]["f1-score"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"]
        }

    def setup_trainer(self, train_dataset, eval_dataset):
        args = TrainingArguments(
            output_dir=str(self.model_output_path),
            num_train_epochs=100,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.log_path),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            dataloader_num_workers=os.cpu_count() - 1 or 2,
            dataloader_pin_memory=False
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def train(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        self.trainer.train()
        #self.trainer.train(resume_from_checkpoint=True) # to resume training if needed


    def evaluate(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        return self.trainer.evaluate()

    def save(self):
        self.model.save_pretrained(self.model_output_path)
        self.tokenizer.save_pretrained(self.model_output_path)
        print(f"[INFO] Model saved to {self.model_output_path}")

    def save_from_latest_checkpoint(self):
        checkpoint_dirs = list(self.model_output_path.glob("checkpoint-*"))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"[ERROR] No checkpoint directories found in {self.model_output_path}")

        latest_checkpoint = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[-1]))
        print(f"[INFO] Found latest checkpoint: {latest_checkpoint}")

        model = RobertaForSequenceClassification.from_pretrained(
            latest_checkpoint,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )

        model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        model.config.save_pretrained(self.model_output_path)

        tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_output_path)

        safe_file = latest_checkpoint / "model.safetensors"
        if safe_file.exists():
            dst_file = self.model_output_path / "pytorch_model.bin"
            shutil.copy(safe_file, dst_file)
            print(f"[INFO] Renamed {safe_file.name} → pytorch_model.bin")
        else:
            raise FileNotFoundError(f"[ERROR] No model.safetensors found in checkpoint {latest_checkpoint}")

        print(f"[INFO] Full model saved from checkpoint {latest_checkpoint.name} to {self.model_output_path}")

    def run(self):
        print("[INFO] Starting CoreTrainer run sequence...")

        self.clean_and_save_dataset()
        split = self.load_and_tokenize_dataset()
        train_dataset = split["train"]
        eval_dataset = split["test"]

        self.setup_trainer(train_dataset, eval_dataset)
        self.train()

        results = self.evaluate()
        print("[RESULTS] Evaluation metrics:", results)

        self.save()
        print("[INFO] CoreTrainer pipeline completed.")
