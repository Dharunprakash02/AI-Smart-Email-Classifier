# models/

Place your trained model files here after running the training pipeline:

- `email_classifier.pkl`    — category LogisticRegression (from train_model.py)
- `vectorizer.pkl`          — category TF-IDF vectoriser
- `urgency_model.pkl`       — urgency LogisticRegression (from train_urgency_model.py)
- `urgency_vectorizer.pkl`  — urgency TF-IDF vectoriser

If your teammate has already trained these, copy them into this folder and run:

    streamlit run app.py
