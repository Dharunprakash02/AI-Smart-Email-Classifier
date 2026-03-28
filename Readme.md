# 📬 AI-Powered Smart Email Classifier

A team project that combines a trained sklearn + NLP pipeline with a professional
Streamlit dashboard for real-time email classification and analytics.

---

## 🗂 Project Structure

```
smart_email_classifier/
│
├── app.py                          ← Streamlit dashboard (entry point)
│
├── src/
│   ├── preprocess.py               ← Download & clean dataset
│   ├── feature_engineering.py      ← TF-IDF vectoriser utilities
│   ├── train_model.py              ← Train category classifier
│   ├── train_urgency_model.py      ← Train urgency classifier
│   ├── create_urgency_labels.py    ← Add rule-based urgency labels to CSVs
│   ├── evaluate_model.py           ← Metrics & confusion matrix helpers
│   ├── predict.py                  ← CLI predictor (standalone)
│   ├── urgency_rule_based.py       ← Keyword-scoring urgency heuristic
│   └── load_hf_dataset.py          ← HuggingFace dataset loader
│
├── models/                         ← Trained model artifacts (pkl)
│   ├── email_classifier.pkl        ← Category LogisticRegression model
│   ├── vectorizer.pkl              ← Category TF-IDF vectoriser
│   ├── urgency_model.pkl           ← Urgency LogisticRegression model
│   └── urgency_vectorizer.pkl      ← Urgency TF-IDF vectoriser
│
├── data/
│   └── processed/                  ← Auto-generated CSVs after preprocessing
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / pull the repo
```bash
git clone <your-repo-url>
cd smart_email_classifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Pipeline (run once, or skip if models already exist)

> If your teammate has already pushed the trained `.pkl` files, jump straight
> to **Running the Dashboard**.

```bash
# Step 1 – download & clean the dataset
python src/preprocess.py

# Step 2 – train the category classifier
python src/train_model.py

# Step 3 – generate rule-based urgency labels
python src/create_urgency_labels.py

# Step 4 – train the urgency classifier
python src/train_urgency_model.py
```

After these steps `models/` will contain four `.pkl` files.

---

## 🚀 Running the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Analyzer** | Classify a single email; view category + urgency with confidence charts |
| **Executive Dashboard** | High-level KPIs, category/urgency distribution, volume trend |
| **Deep Analytics** | Weekday-hour heatmap, urgency load area chart, keyword frequency, Pareto |
| **Record Explorer** | Searchable, sortable log of all classified emails with full drilldown |
| **Batch CSV Analyzer** | Upload a CSV → classify hundreds of emails → download results |

---

## 🔬 CLI Prediction (no UI needed)

```bash
python src/predict.py
# or pass text directly:
python src/predict.py --text "Our server is down, fix it ASAP!"
```

---

## 📦 Model Details

| Model | Algorithm | Input | Output |
|-------|-----------|-------|--------|
| `email_classifier.pkl` | Logistic Regression | TF-IDF (5 000 features) | 6 categories |
| `urgency_model.pkl` | Logistic Regression (balanced) | TF-IDF (5 000 features) | Low / Medium / High |

Categories: **Academic · Complaint · Request · Feedback · Spam · General**

Urgency detection uses a **hybrid approach**: rule-based keyword scoring takes
priority for clearly urgent language; the ML model handles ambiguous cases.

---

## 👥 Team

| Role | Deliverable |
|------|-------------|
| Data & Training | `src/` pipeline scripts + `models/*.pkl` |
| Dashboard | `app.py` Streamlit UI |

---

## 📝 Notes

- `email_predictions.csv` is auto-created in the project root on first run.
- The Streamlit model cache is cleared automatically whenever a new prediction is saved.
- PDF report export requires `fpdf2` (included in `requirements.txt`).
