import sys
from pathlib import Path
import traceback

from AutoExtractor.model import AutoExtractor
from ModelTrainer.CoreTrainer import CoreTrainer
from ModelComparison.compare.compare_model import internal_evaluate


def run_extraction():
    print("📦 Starting data extraction...")
    try:
        extractor = AutoExtractor()
        extractor.extract_all()
        print("✅ Data extraction complete.")
    except Exception as e:
        print("❌ Data extraction failed:")
        traceback.print_exc()


def run_training():
    print("🧠 Starting model training...")
    try:
        trainer = CoreTrainer()
        trainer.clean_and_save_dataset()
        trainer.load_and_tokenize_dataset()
        #trainer.run()
        #trainer.save_from_latest_checkpoint()
        print("✅ Training complete and model saved.")
    except Exception as e:
        print("❌ Model training failed:")
        traceback.print_exc()


def run_model_comparison(test_data_path: Path, models: list = None, force_download: bool = False):
    print(f"📊 Starting model comparison using dataset: {test_data_path}")
    try:
        internal_evaluate(dataset_path=str(test_data_path), models=models, force_download=force_download)
        print("✅ Model comparison complete.")
    except Exception as e:
        print("❌ Model comparison failed:")
        traceback.print_exc()


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    test_csv_path = base_path / "dataset" / "training" / "Core" / "test.csv"

    run_extraction()
    run_training()
    run_model_comparison(test_data_path=test_csv_path)

