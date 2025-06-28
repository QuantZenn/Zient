# 🧠 Zient

Zient is a modular framework for extracting financial data, training sentiment models, and comparing them against industry-standard LLMs (like LLaMa, Mixtral, FinBERT, etc.). Designed for flexibility in fine-tuning and batch evaluation, Zient is intended for financial sentiment experimentation at scale.

---

## 📁 Project Structure

```
Zient/
│
├── meta/                  # Metadata or static references
├── models/                # Saved models and checkpoints
├── outputs/               # Prediction results
└── src/
    ├── AutoExtractor/
    │   ├── extractors/
    │   │   └── format_extractor.py
    │   └── model/
    │       └── AutoExtractor.py
    │
    ├── registry/
    │   ├── FormatRegistry.py
    │   └── formats.json
    │
    ├── ModelComparison/
    │   ├── compare/
    │   │   └── compare_model.py
    │   ├── config/
    │   │   ├── evn.py
    │   │   └── model_cmp.py
    │   ├── Model/
    │   │   └── ModelComparison.py
    │   └── ModelWrappers/
    │       ├── ChatGPTWrapper.py
    │       ├── CommandRPlusWrapper.py
    │       ├── DeepSeekWrapper.py
    │       ├── FinBERTWrapper.py
    │       ├── GemmaWrapper.py
    │       ├── LLaMaWrapper.py
    │       ├── MistralWrapper.py
    │       ├── MixtralWrapper.py
    │       └── ZientCoreWrapper.py
    │
    └── ModelTrainer/
        ├── __init__.py
        └── CoreTrainer.py

    main.py             # Entry point for running extraction/training/evaluation
```

---

## 🚀 How to Run

### 1. Extract Data

```python
# In main.py
extractor = AutoExtractor()
extractor.extract_all()
```

### 2. Train the Model

```python
trainer = CoreTrainer()
trainer.run()

# Optional: generate and save model from latest checkpoint
# if you didn't finish training previously
# trainer.save_from_latest_checkpoint()
```

### 3. Run Model Comparison

```python
from ModelComparison.compare.compare_model import internal_evaluate

internal_evaluate(
    dataset_path="path/to/test.csv",
    models=None  # or provide specific model names
)
```

📄 The results will be saved in:

```
outputs/models/{model_name}_predictions.csv
```

---

## 💡 Notes

* Large models like LLaMa, Mixtral, etc., are cached in `llm_cache/` and **excluded** from GitHub.
* Git LFS is used for managing `pytorch_model.bin` files (if required).
* Ensure `test.csv` uses human-readable labels: `positive`, `neutral`, `negative`.

---

## 📌 Future Work

* Add sector/ticker-specific fine-tuning support.
* Upload final models and comparison logs to Hugging Face.

---

## 📟 License

(Include your license type here)

---

## ✍️ Author

QuantZenn
GitHub: [@QuantZenn](https://github.com/QuantZenn/Zient/)
