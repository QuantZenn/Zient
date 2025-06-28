# 🧠 Zient — Financial Sentiment Analysis Framework

Zient is an end-to-end sentiment analysis framework designed for financial news and data. It supports automated data extraction, model training, and evaluation against various LLMs. 

---

## 📦 Project Structure

Zient/
├── meta/ # Metadata or auxiliary files
├── models/ # Trained models and checkpoints
├── outputs/ # Prediction outputs and evaluation results
├── src/
│ ├── AutoExtractor/ # Raw financial data extractors
│ │ ├── extractors/
│ │ │ └── format_extractor.py
│ │ └── model/
│ │ └── AutoExtractor.py
│ ├── registry/ # Format registry and definitions
│ │ ├── FormatRegistry.py
│ │ ├── formats.json
│ ├── ModelTrainer/ # Custom model training pipeline
│ │ ├── init.py
│ │ └── CoreTrainer.py
│ ├── ModelComparison/ # Evaluation pipeline and model wrappers
│ │ ├── compare/
│ │ │ └── compare_model.py
│ │ ├── config/
│ │ │ └── model_cmp.py, evn.py
│ │ ├── Model/
│ │ │ └── ModelComparison.py
│ │ └── ModelWrappers/
│ │ ├── FinBERTWrapper.py
│ │ ├── ChatGPTWrapper.py
│ │ ├── DeepSeekWrapper.py
│ │ ├── ZientCoreWrapper.py
│ │ ├── LLaMaWrapper.py
│ │ ├── MistralWrapper.py
│ │ ├── MixtralWrapper.py
│ │ ├── CommandRPlusWrapper.py
│ │ └── GemmaWrapper.py
│ └── main.py # Entrypoint for training and evaluation
├── .env
└── README.md
└── requiremenr.txt

---

## ⚙️ Components

### ✅ AutoExtractor
Extracts and standardizes financial news data into a unified format for training.

- `AutoExtractor.py`: Main class
- `format_extractor.py`: Handles different source formats (e.g., CSV, JSON, TXT, DF)

### ✅ ModelTrainer
Handles model training from cleaned/structured financial data.

- `CoreTrainer.py`: Trains your in-house sentiment classification model (ZientCore)

### ✅ ModelComparison
Evaluates your model against other open-source and API-based LLMs.

Includes wrappers for:

- 🤖 `FinBERT`
- 🤖 `ChatGPT`
- 🤖 `ZientCore`
- 🤖 `DeepSeek`
- 🤖 `LLaMa`
- 🤖 `Mistral`
- 🤖 `Mixtral`
- 🤖 `Command-R+`
- 🤖 `Gemma`

Each wrapper implements `predict_batch()` and supports CSV or dataframe input.

---

## 🚀 How to Run

### 1. Extract Data
```bash
# In main.py
extractor = AutoExtractor()
extractor.extract_all()

2. Train the Model
trainer = CoreTrainer()
trainer.run()
#trainer.save_from_latest_checkpoint()  # Optional: generate and save the model from the latest checkpoint if training completed partially and you want to avoid retraining

3. Run Model Comparison
from ModelComparison.compare.compare_model import internal_evaluate

internal_evaluate(dataset_path="path/to/test.csv", models=None)
The result will be saved in outputs/models/{model_name}_predictions.csv

🧼 Notes
Large models like LLaMa, Mixtral, etc., are cached in llm_cache/ and excluded from git using .gitignore.

Git LFS is used for managing pytorch_model.bin files (if required).

Ensure test.csv uses human-readable labels (positive, neutral, negative) or the internal logic handles conversion from 0/1/2.

📌 Future Work
Add sector/ticker-specific fine-tuning


📜 License


✍️ Author
...
GitHub: @QuantZennq