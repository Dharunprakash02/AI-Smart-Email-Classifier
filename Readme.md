# AI Smart Email Classifier

An intelligent email classification system that categorizes emails and predicts their urgency level using DistilBERT transformer models.

## Project Structure

```
├── app.py                          # Main Streamlit application
├── models/
│   ├── category_model/             # Email category classification model
│   ├── urgency_model/              # Urgency prediction model
│   └── tokenizer/                  # Shared BERT tokenizer
├── email_predictions.csv           # Output predictions file
├── requirements.txt                # Python dependencies
└── .venv                          # Virtual environment
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Activate Virtual Environment

On Windows PowerShell:
```powershell
& ".venv\Scripts\Activate.ps1"
```

On Windows Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Once the virtual environment is activated and dependencies are installed, run:

```bash
python -m streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Features

- **Email Classification**: Automatically categorizes emails into 6 categories:
  - Complaint
  - Request
  - Feedback
  - Spam
  - Inquiry
  - Other

- **Urgency Detection**: Predicts email urgency level:
  - Low
  - Medium
  - High

- **Interactive Dashboard**: Visualize predictions and statistics with interactive charts

## Models

The project uses two pre-trained DistilBERT models:
- `category_model`: Fine-tuned for email category classification
- `urgency_model`: Fine-tuned for urgency prediction
- `tokenizer`: BERT tokenizer for text preprocessing

### Important: Large Model Files

The model files (`.safetensors`, `.zip`, and `training_args.bin`) are not included in this repository due to their large size (>100MB each).

**How to get the model files:**

1. **Option 1: Manual Download**
   - Download the pre-trained models from your model hosting service
   - Extract them into the `models/` directory maintaining the structure:
     ```
     models/
     ├── category_model/
     │   ├── config.json
     │   ├── model.safetensors
     │   └── training_args.bin
     ├── urgency_model/
     │   ├── config.json
     │   ├── model.safetensors
     │   └── training_args.bin
     └── tokenizer/
         ├── tokenizer.json
         └── tokenizer_config.json
     ```

2. **Option 2: Using Git LFS (Recommended for future projects)**
   - Install Git LFS: `git lfs install`
   - Track large files: `git lfs track "**/*.safetensors" "**/*.zip"`
   - Add and commit: `git add .gitattributes && git commit -m "Setup LFS"`