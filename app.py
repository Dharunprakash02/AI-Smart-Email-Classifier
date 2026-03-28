import os
import re
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Email Intelligence Dashboard",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE = "email_predictions.csv"

CATEGORY_LABELS = ["Complaint", "Feedback", "Enquiry", "Spam", "Request", "Others"]
CATEGORY_LABELS_LOWER = [c.lower() for c in CATEGORY_LABELS]

CATEGORY_NORMALIZATION_MAP = {
    "complaint": "Complaint",
    "feedback": "Feedback",
    "enquiry": "Enquiry",
    "inquiry": "Enquiry",
    "spam": "Spam",
    "request": "Request",
    "other": "Others",
    "others": "Others",
    "general": "Others",
    "academic": "Enquiry",
    # Native HF dataset classes handled by trained model artifacts.
    "forum": "Enquiry",
    "promotions": "Spam",
    "social_media": "Others",
    "updates": "Feedback",
    "verify_code": "Request",
}


def normalize_category_label(value: str) -> str:
    key = str(value).strip().lower()
    return CATEGORY_NORMALIZATION_MAP.get(key, "Others")


def rule_based_category(text: str):
    t = str(text).lower()
    rules = {
        "Spam": [
            "free", "winner", "claim", "click", "offer", "limited time",
            "suspend", "deactivate", "verify account", "lottery", "congratulations",
        ],
        "Complaint": [
            "not working", "issue", "problem", "failed", "error", "angry",
            "frustrated", "bad service", "refund", "charged", "delay",
            "slow", "slowly", "slowness", "performance", "responding slowly",
            "affecting productivity", "disruption", "outage", "resolve the issue",
        ],
        "Feedback": [
            "feedback", "suggest", "improve", "great", "excellent", "appreciate",
            "experience", "review", "feature request",
        ],
        "Request": [
            "please", "kindly", "could you", "can you", "i need", "request",
            "help me", "grant", "approve", "provide", "share",
        ],
        "Enquiry": [
            "enquiry", "inquiry", "question", "clarify", "details", "how to",
            "what is", "when is", "where is", "why is",
        ],
    }
    scores = {k: 0 for k in rules}
    for label, keywords in rules.items():
        scores[label] = sum(1 for kw in keywords if kw in t)

    # Question-like phrasing is typically an enquiry.
    if "?" in t:
        scores["Enquiry"] += 1

    # Strong intent hints.
    if "could you" in t or "please" in t or "kindly" in t:
        scores["Request"] += 1
    if "not satisfied" in t or "doesn't work" in t or "does not work" in t:
        scores["Complaint"] += 1

    # Polite language can appear inside complaints; do not let it overpower issue signals.
    if scores["Complaint"] >= 2 and ("please" in t or "kindly" in t):
        scores["Complaint"] += 1
        scores["Request"] = max(0, scores["Request"] - 1)

    # Operational pain signals are stronger complaint indicators than generic requests.
    complaint_context_terms = [
        "affecting productivity", "downtime", "slow", "slowly", "issue", "problem",
        "error", "failed", "not working", "outage", "investigate",
    ]
    if any(term in t for term in complaint_context_terms):
        scores["Complaint"] += 1

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score == 0:
        return "Others", 0, scores
    return best_label, best_score, scores

URGENCY_LABELS   = ["Low", "Medium", "High"]
URGENCY_PRIORITY = ["High", "Medium", "Low"]

URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "not working",
    "server down", "critical", "error", "emergency",
    "system down", "deadline",
]

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "your",
    "please", "would", "there", "their", "about", "they", "them", "were",
    "what", "when", "where", "which", "while", "will", "into", "just",
    "need", "hello", "regards", "thanks", "thank", "subject", "email",
    "message", "team", "dear",
}

# ─── Styles ────────────────────────────────────────────────────────────────────
def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(900px 400px at 10% -10%, #0b1220 0%, transparent 55%),
                    radial-gradient(700px 350px at 95% 0%, #0a1320 0%, transparent 50%),
                    linear-gradient(180deg, #030712 0%, #000000 100%);
                color: #e5e7eb;
            }
            .stMarkdown, .stMarkdown p, .stMarkdown li,
            .st-emotion-cache-10trblm, [data-testid="stSidebar"] * {
                color: #e5e7eb;
            }
            .hero {
                background: linear-gradient(125deg, #0f172a 0%, #134e4a 100%);
                border-radius: 16px;
                padding: 1.2rem 1.4rem;
                border: 1px solid rgba(255,255,255,0.16);
                margin-bottom: 1rem;
            }
            .hero h1 { margin:0; color:#ffffff; font-size:1.9rem; font-weight:700; }
            .hero p  { margin:0.4rem 0 0; color:#dbeafe; font-size:0.95rem; }
            [data-testid="stMetric"] {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 0.4rem 0.6rem;
                box-shadow: 0 8px 28px rgba(0,0,0,0.35);
            }
            .stTabs [data-baseweb="tab-list"] { gap:0.45rem; padding-bottom:0.2rem; }
            .stTabs [data-baseweb="tab"] {
                background:#1f2937; border-radius:10px;
                color:#d1d5db; font-weight:600; padding:0.45rem 0.9rem;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(130deg,#0e7490 0%,#0284c7 100%);
                color:#ffffff;
            }
            .section-caption { color:#94a3b8; font-size:0.92rem; margin-bottom:0.9rem; }
            .result-box {
                border:1px solid #334155; background:#111827;
                border-radius:12px; padding:0.85rem 1rem;
                margin-top:0.8rem; color:#e2e8f0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Data helpers ──────────────────────────────────────────────────────────────
def ensure_data_file(path: str) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=["date", "email", "category", "urgency"]).to_csv(path, index=False)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    ensure_data_file(path)
    df = pd.read_csv(path)
    for col in ["date", "email", "category", "urgency"]:
        if col not in df.columns:
            df[col] = ""
    df = df[["date", "email", "category", "urgency"]]
    if df.empty:
        return df
    df["category"] = df["category"].apply(normalize_category_label)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def save_prediction(record: dict, path: str) -> None:
    ensure_data_file(path)
    try:
        current_df = pd.read_csv(path)
    except Exception:
        current_df = pd.DataFrame(columns=["date", "email", "category", "urgency"])
    record = dict(record)
    record["category"] = normalize_category_label(record.get("category", ""))
    updated_df = pd.concat([current_df, pd.DataFrame([record])], ignore_index=True)
    updated_df.to_csv(path, index=False)
    load_data.clear()


# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Loads the trained sklearn (pickle) models.
    Expected files inside models/:
        email_classifier.pkl      – category LogisticRegression
        vectorizer.pkl            – category TF-IDF vectorizer
        urgency_model.pkl         – urgency LogisticRegression
        urgency_vectorizer.pkl    – urgency TF-IDF vectorizer
    """
    import pickle, nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

    base = os.path.join(os.path.dirname(__file__), "models")

    def _load(name):
        p = os.path.join(base, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file not found: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    category_model     = _load("email_classifier.pkl")
    category_vec       = _load("vectorizer.pkl")
    urgency_model      = _load("urgency_model.pkl")
    urgency_vec        = _load("urgency_vectorizer.pkl")

    return category_model, category_vec, urgency_model, urgency_vec


# ─── Text cleaning (mirrors preprocess.py) ─────────────────────────────────────
def clean_text(text: str) -> str:
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    if not isinstance(text, str):
        return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# ─── Rule-based urgency (mirrors urgency_rule_based.py) ────────────────────────
def rule_based_urgency(text: str):
    t = text.lower()
    high_kw   = ["urgent","asap","immediately","critical","system down","emergency","deadline"]
    medium_kw = ["soon","priority","update","review","request","schedule","meeting","approval"]
    score = sum(2 for w in high_kw if w in t) + sum(1 for w in medium_kw if w in t)
    if score >= 2:  return "High"
    if score == 1:  return "Medium"
    return "Low"


# ─── Core prediction ───────────────────────────────────────────────────────────
CATEGORY_MAPPING = {
    # Model IDs are trained on: forum, promotions, social_media, spam, updates, verify_code
    0: "Enquiry", 1: "Spam", 2: "Others",
    3: "Spam",    4: "Others", 5: "Request",
}

def predict_email(raw_text: str) -> dict:
    cat_model, cat_vec, urg_model, urg_vec = load_models()
    cleaned = clean_text(raw_text)

    # ── Category ──────────────────────────────────────────────────────────────
    cat_tfidf   = cat_vec.transform([cleaned])
    cat_pred    = cat_model.predict(cat_tfidf)[0]
    cat_proba   = cat_model.predict_proba(cat_tfidf)[0]
    cat_classes = cat_model.classes_

    # Supports both legacy numeric-class models and newer string-label models.
    if isinstance(cat_pred, (int, float)):
        model_category = normalize_category_label(CATEGORY_MAPPING.get(int(cat_pred), "Others"))
    else:
        model_category = normalize_category_label(str(cat_pred))

    # Build full distribution (by label name)
    category_distribution = {}
    for idx, cls in enumerate(cat_classes):
        if isinstance(cls, (int, float)):
            label = normalize_category_label(CATEGORY_MAPPING.get(int(cls), str(cls)))
        else:
            label = normalize_category_label(str(cls))
        category_distribution[label] = category_distribution.get(label, 0.0) + float(cat_proba[idx])

    # Pad any missing labels with 0
    for lbl in CATEGORY_LABELS:
        category_distribution.setdefault(lbl, 0.0)

    # Hybrid category decision: use rule signal when model is generic/uncertain.
    model_confidence_for_model_category = float(category_distribution.get(model_category, 0.0))

    rule_category, rule_score, rule_scores = rule_based_category(raw_text)
    if rule_score >= 2:
        category = rule_category
        category_source = "rule"
    elif rule_scores.get("Complaint", 0) >= 2 and rule_scores.get("Complaint", 0) >= rule_scores.get("Request", 0):
        category = "Complaint"
        category_source = "hybrid"
    elif rule_score == 1 and (model_confidence_for_model_category < 0.62 or model_category in ("Others", "Enquiry")):
        category = rule_category
        category_source = "hybrid"
    else:
        category = model_category
        category_source = "model"

    # Confidence should reflect the chosen category, not just max raw class score.
    model_confidence_for_chosen = float(category_distribution.get(category, 0.0))
    if category_source == "model":
        category_confidence = model_confidence_for_chosen
    elif category_source == "hybrid":
        category_confidence = max(model_confidence_for_chosen, min(0.85, 0.55 + 0.08 * rule_score))
    else:
        category_confidence = max(model_confidence_for_chosen, min(0.95, 0.60 + 0.10 * rule_score))

    # Blend a little rule signal into displayed distribution for better calibration.
    rule_total = sum(rule_scores.values())
    if rule_total > 0:
        blend = 0.18
        for lbl in CATEGORY_LABELS:
            rule_p = rule_scores.get(lbl, 0) / rule_total
            category_distribution[lbl] = (1 - blend) * category_distribution.get(lbl, 0.0) + blend * rule_p

        # Re-normalize after blending.
        total_p = sum(category_distribution.values())
        if total_p > 0:
            for lbl in CATEGORY_LABELS:
                category_distribution[lbl] = float(category_distribution[lbl] / total_p)

    # ── Urgency ───────────────────────────────────────────────────────────────
    rule_urg = rule_based_urgency(raw_text)

    # Check if rule fires strongly (High)
    if rule_urg == "High":
        urgency          = "High"
        urgency_confidence = 1.0
        urgency_source   = "rule"
        urgency_distribution = {"Low": 0.0, "Medium": 0.0, "High": 1.0}
    else:
        urg_tfidf    = urg_vec.transform([cleaned])
        urg_label    = str(urg_model.predict(urg_tfidf)[0])
        urg_proba    = urg_model.predict_proba(urg_tfidf)[0]
        urg_classes  = urg_model.classes_

        # Normalise label casing
        label_map = {c.lower(): c for c in ["Low","Medium","High"]}
        urgency   = label_map.get(urg_label.lower(), urg_label)

        urgency_confidence   = float(max(urg_proba))
        urgency_source       = "model"
        urgency_distribution = {}
        for idx, cls in enumerate(urg_classes):
            norm = label_map.get(str(cls).lower(), str(cls))
            urgency_distribution[norm] = float(urg_proba[idx])
        for lbl in ["Low","Medium","High"]:
            urgency_distribution.setdefault(lbl, 0.0)

    return {
        "category":              category,
        "urgency":               urgency,
        "category_confidence":   category_confidence,
        "urgency_confidence":    urgency_confidence,
        "urgency_source":        urgency_source,
        "category_distribution": category_distribution,
        "urgency_distribution":  urgency_distribution,
    }


# ─── Recommendation ───────────────────────────────────────────────────────────
def get_recommendation(category: str, urgency: str) -> str:
    if urgency.lower() == "high":
        return "🚨 Escalate to operations immediately and assign an owner."
    if category.lower() == "complaint":
        return "🎫 Open a support ticket and respond with a resolution timeline."
    if category.lower() == "request":
        return "� Assign to the service desk queue and confirm expected turnaround."
    if category.lower() == "spam":
        return "� Flag as spam and exclude from all action queues."
    if category.lower() == "feedback":
        return "� Log feedback and route to the product / CX team for review."
    return "📥 Route to standard triage for follow-up."


# ─── PDF report ───────────────────────────────────────────────────────────────
def build_report_pdf(report: dict):
    try:
        from fpdf import FPDF
    except Exception:
        return None, "PDF export requires fpdf2. Install with: pip install fpdf2"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def safe_cell(text, lh=6):
        # fpdf core fonts are Latin-1 only; replace unsupported chars (e.g. emoji).
        safe_text = str(text).encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, lh, safe_text, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 16)
    safe_cell("Email Classification Report", 10)
    pdf.set_font("Helvetica", "", 10)
    safe_cell(f"Generated: {report['timestamp']}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    safe_cell("Prediction Summary", 8)
    pdf.set_font("Helvetica", "", 10)
    safe_cell(f"Category : {report['category']} ({report['category_confidence']:.1%})")
    safe_cell(f"Urgency  : {report['urgency']} ({report['urgency_confidence']:.1%})")
    safe_cell(f"Source   : {report['urgency_source']}")
    safe_cell(f"Action   : {report['recommendation']}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    safe_cell("Category Probabilities", 8)
    pdf.set_font("Helvetica", "", 10)
    for lbl, val in sorted(report["category_distribution"].items(), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 6, f"- {lbl}: {val:.1%}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    safe_cell("Urgency Probabilities", 8)
    pdf.set_font("Helvetica", "", 10)
    for lbl, val in sorted(report["urgency_distribution"].items(), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 6, f"- {lbl}: {val:.1%}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    safe_cell("Email Content", 8)
    pdf.set_font("Helvetica", "", 9)
    preview = report["email"][:1800] + ("..." if len(report["email"]) > 1800 else "")
    safe_cell(preview, 5)

    return bytes(pdf.output(dest="S")), None


# ─── Render report section ────────────────────────────────────────────────────
def render_prediction_report(report: dict) -> None:
    st.markdown("### 📊 Analysis Report")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Category", report["category"])
    m2.metric("Predicted Urgency",  report["urgency"])
    m3.metric("Urgency Source",     report["urgency_source"].title())

    s1, s2 = st.columns(2)
    s1.metric("Category Confidence", f"{report['category_confidence']:.1%}")
    s2.metric("Urgency Confidence",  f"{report['urgency_confidence']:.1%}")

    st.info(report["recommendation"])

    cat_df = (
        pd.DataFrame([{"category": k, "probability": v}
                      for k, v in report["category_distribution"].items()])
        .sort_values("probability", ascending=True)
        .reset_index(drop=True)
    )
    urg_df = pd.DataFrame([{"urgency": k, "probability": v}
                            for k, v in report["urgency_distribution"].items()])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            cat_df, x="probability", y="category", orientation="h",
            title="Category Probability Breakdown",
            text=cat_df["probability"].map(lambda v: f"{v:.1%}"),
            color="probability", color_continuous_scale="Teal",
        )
        fig.update_layout(showlegend=False, xaxis_title="Probability", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with c2:
        urgency_color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        fig2 = go.Figure(data=[go.Pie(
            labels=urg_df["urgency"], values=urg_df["probability"],
            hole=0.55,
            marker=dict(colors=[urgency_color_map.get(u, "#64748b") for u in urg_df["urgency"]]),
            textinfo="label+percent",
        )])
        fig2.update_layout(title="Urgency Probability Breakdown")
        st.plotly_chart(fig2, width="stretch")

    high_pct = report["urgency_distribution"].get("High", 0.0) * 100
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number", value=high_pct, number={"suffix": "%"},
        title={"text": "High Urgency Risk"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "#ef4444"},
            "steps": [
                {"range": [0, 35],  "color": "#d1fae5"},
                {"range": [35, 70], "color": "#fef3c7"},
                {"range": [70,100], "color": "#fee2e2"},
            ],
        },
    ))
    fig3.update_layout(height=280)
    st.plotly_chart(fig3, width="stretch")

    pdf_bytes, pdf_error = build_report_pdf(report)
    if pdf_bytes:
        st.download_button(
            "⬇️ Download PDF Report", data=BytesIO(pdf_bytes),
            file_name=f"email_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", width="stretch",
        )
    else:
        st.warning(pdf_error)


# ─── Filter helper ────────────────────────────────────────────────────────────
def apply_filters(df, date_range, sel_cats, sel_urgs, search):
    if df.empty:
        return df
    fdf = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s, e = date_range
        fdf = fdf[(fdf["date"].dt.date >= s) & (fdf["date"].dt.date <= e)]
    if sel_cats:
        fdf = fdf[fdf["category"].isin(sel_cats)]
    else:
        return fdf.iloc[0:0]
    if sel_urgs:
        fdf = fdf[fdf["urgency"].isin(sel_urgs)]
    else:
        return fdf.iloc[0:0]
    if search.strip():
        fdf = fdf[fdf["email"].str.contains(search.strip(), case=False, na=False)]
    return fdf


# ─── Keyword extractor ────────────────────────────────────────────────────────
def extract_top_keywords(email_series: pd.Series, top_n: int = 12) -> pd.DataFrame:
    words = []
    for text in email_series.fillna(""):
        tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        words.extend(t for t in tokens if t not in STOPWORDS)
    if not words:
        return pd.DataFrame(columns=["keyword", "count"])
    kdf = pd.Series(words).value_counts().head(top_n).reset_index()
    kdf.columns = ["keyword", "count"]
    return kdf


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    render_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>📬 Smart Email Intelligence Dashboard</h1>
            <p>AI-powered classification · urgency detection · operational analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data(DATA_FILE)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔎 Filters")

        if df.empty:
            st.info("No records yet. Analyze an email in the Analyzer tab to begin.")
            filtered_df = df
        else:
            quick = st.selectbox(
                "Quick filter",
                ["Custom", "High urgency only", "Complaints only", "Last 7 days"],
            )
            min_d, max_d = df["date"].min().date(), df["date"].max().date()
            date_range = st.date_input(
                "Date range",
                value=(max(min_d, max_d - timedelta(days=30)), max_d),
                min_value=min_d, max_value=max_d,
            )

            avail_cats = sorted(set(CATEGORY_LABELS) | set(df["category"].dropna().unique()))
            cat_choice = st.selectbox("Category", ["All"] + avail_cats)
            sel_cats   = avail_cats if cat_choice == "All" else [cat_choice]

            avail_urgs = [u for u in URGENCY_PRIORITY if u in set(df["urgency"].dropna().unique())]
            urg_choice = st.selectbox("Urgency", ["All"] + URGENCY_PRIORITY)
            sel_urgs   = avail_urgs if urg_choice == "All" else [urg_choice]

            search    = st.text_input("Search email text", placeholder="Type keywords…")
            filtered_df = apply_filters(df, date_range, sel_cats, sel_urgs, search)

            if quick == "High urgency only":
                filtered_df = filtered_df[filtered_df["urgency"] == "High"]
            elif quick == "Complaints only":
                filtered_df = filtered_df[filtered_df["category"] == "Complaint"]
            elif quick == "Last 7 days":
                cutoff = datetime.now().date() - timedelta(days=7)
                filtered_df = filtered_df[filtered_df["date"].dt.date >= cutoff]

            st.divider()
            st.metric("Filtered records", len(filtered_df))
            st.metric("Total records",    len(df))

            if not filtered_df.empty:
                st.download_button(
                    "⬇️ Export filtered CSV",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"filtered_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", width="stretch",
                )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Analyzer",
        "📈 Executive Dashboard",
        "🔍 Deep Analytics",
        "📋 Record Explorer",
        "📂 Batch CSV Analyzer",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 – ANALYZER
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Classify New Email")
        st.markdown(
            '<p class="section-caption">Paste email text → run AI classification → result is logged to your dataset.</p>',
            unsafe_allow_html=True,
        )

        col_form, col_stats = st.columns([3, 1.2])

        with col_form:
            with st.form("email_form", clear_on_submit=False):
                email_text = st.text_area(
                    "Email content", height=220,
                    placeholder="Paste the email text here…",
                )
                submitted = st.form_submit_button("🔍 Analyze Email", width="stretch", type="primary")

        with col_stats:
            wc = len(email_text.split()) if isinstance(email_text, str) and email_text else 0
            cc = len(email_text)         if isinstance(email_text, str) else 0
            rt = max(1, round(wc / 200)) if wc else 0
            st.metric("Word count",  wc)
            st.metric("Characters",  cc)
            st.metric("Read time",   f"{rt} min" if rt else "0 min")

        if submitted:
            if not email_text.strip():
                st.warning("Please enter email text before analyzing.")
            else:
                try:
                    with st.status("Running classification pipeline…", expanded=True) as status:
                        status.write("⚙️ Loading models from disk (cached after first load)…")
                        result = predict_email(email_text)
                        status.write("💾 Saving prediction to dataset…")
                        report = {
                            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "email":                 email_text,
                            "category":              result["category"],
                            "urgency":               result["urgency"],
                            "category_confidence":   result["category_confidence"],
                            "urgency_confidence":    result["urgency_confidence"],
                            "urgency_source":        result["urgency_source"],
                            "category_distribution": result["category_distribution"],
                            "urgency_distribution":  result["urgency_distribution"],
                            "recommendation":        get_recommendation(result["category"], result["urgency"]),
                        }
                        save_prediction(
                            {"date": datetime.now(), "email": email_text,
                             "category": result["category"], "urgency": result["urgency"]},
                            DATA_FILE,
                        )
                        st.session_state["latest_report"] = report
                        status.update(label="✅ Email analyzed successfully", state="complete")

                    st.success("Prediction completed and saved to dataset.")
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

        latest = st.session_state.get("latest_report")
        if latest:
            render_prediction_report(latest)
            with st.expander("📄 Submitted email content"):
                st.write(latest.get("email", ""))

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 – EXECUTIVE DASHBOARD
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Executive Dashboard")
        st.markdown('<p class="section-caption">Volume, urgency pressure, and category composition at a glance.</p>',
                    unsafe_allow_html=True)

        if filtered_df.empty:
            st.info("No records match the selected filters.")
        else:
            total   = len(filtered_df)
            hi_urg  = int((filtered_df["urgency"] == "High").sum())
            cmp_cnt = int((filtered_df["category"] == "Complaint").sum())
            days    = max(1, filtered_df["date"].dt.date.nunique())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Emails",       total)
            m2.metric("High Urgency",       hi_urg,  delta=f"{hi_urg/total:.1%}")
            m3.metric("Complaint Share",    f"{cmp_cnt/total*100:.1f}%")
            m4.metric("Avg Emails / Day",   f"{total/days:.1f}")

            c1, c2 = st.columns([1.35, 1])
            with c1:
                cat_cnt = filtered_df["category"].value_counts().reset_index()
                cat_cnt.columns = ["category","count"]
                fig = px.bar(
                    cat_cnt.sort_values("count", ascending=True),
                    x="count", y="category", orientation="h", text="count",
                    color="category", color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Category Distribution",
                )
                fig.update_layout(showlegend=False, xaxis_title="Emails", yaxis_title="")
                st.plotly_chart(fig, width="stretch")

            with c2:
                urg_cnt = filtered_df["urgency"].value_counts().reindex(URGENCY_PRIORITY, fill_value=0)
                fig2 = go.Figure(data=[go.Pie(
                    labels=urg_cnt.index, values=urg_cnt.values, hole=0.62,
                    marker=dict(colors=["#ef4444","#f59e0b","#10b981"]),
                    textinfo="label+percent",
                )])
                fig2.update_layout(title="Urgency Composition", margin=dict(t=50,b=20,l=20,r=20))
                st.plotly_chart(fig2, width="stretch")

            st.divider()

            trend = (
                filtered_df.assign(day=filtered_df["date"].dt.date)
                .groupby("day").size().reset_index(name="count")
            )
            trend["day"] = pd.to_datetime(trend["day"])
            trend["ma7"] = trend["count"].rolling(7, min_periods=1).mean()

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=trend["day"], y=trend["count"],
                mode="lines+markers", name="Daily volume",
                line=dict(color="#0f766e", width=2.2)))
            fig3.add_trace(go.Scatter(x=trend["day"], y=trend["ma7"],
                mode="lines", name="7-day MA",
                line=dict(color="#0284c7", width=2, dash="dot")))
            fig3.update_layout(title="Email Volume Trend", xaxis_title="Date",
                               yaxis_title="Count", hovermode="x unified")
            st.plotly_chart(fig3, width="stretch")

            ct = pd.crosstab(filtered_df["category"], filtered_df["urgency"])
            ct = ct.reindex(columns=URGENCY_PRIORITY, fill_value=0)
            fig4 = px.imshow(ct, text_auto=True, aspect="auto",
                color_continuous_scale="Teal", title="Category × Urgency Heatmap",
                labels={"x":"Urgency","y":"Category","color":"Emails"})
            st.plotly_chart(fig4, width="stretch")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 – DEEP ANALYTICS
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Deep Analytics")
        st.markdown('<p class="section-caption">Behavioral patterns and operational risk indicators.</p>',
                    unsafe_allow_html=True)

        if filtered_df.empty:
            st.info("No records available.")
        else:
            tdf = filtered_df.copy()
            tdf["weekday"] = tdf["date"].dt.day_name()
            tdf["hour"]    = tdf["date"].dt.hour

            wday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pivot = pd.pivot_table(
                tdf, index="weekday", columns="hour",
                values="email", aggfunc="count", fill_value=0,
            ).reindex(wday_order)

            a1, a2 = st.columns(2)
            with a1:
                fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlGnBu",
                    title="Weekday-Hour Traffic Heatmap",
                    labels={"x":"Hour","y":"Weekday","color":"Emails"})
                st.plotly_chart(fig, width="stretch")

            with a2:
                urg_daily = (
                    tdf.assign(day=tdf["date"].dt.date)
                    .groupby(["day","urgency"]).size().reset_index(name="count")
                )
                fig2 = px.area(urg_daily, x="day", y="count", color="urgency",
                    title="Urgency Load Over Time",
                    color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"})
                fig2.update_layout(xaxis_title="Date", yaxis_title="Email count")
                st.plotly_chart(fig2, width="stretch")

            b1, b2 = st.columns(2)
            with b1:
                kdf = extract_top_keywords(tdf["email"], top_n=12)
                if kdf.empty:
                    st.info("Not enough data for keyword extraction.")
                else:
                    fig3 = px.bar(kdf.sort_values("count", ascending=True),
                        x="count", y="keyword", orientation="h",
                        color="count", color_continuous_scale="Blues",
                        title="Most Frequent Terms")
                    fig3.update_layout(showlegend=False, xaxis_title="Mentions", yaxis_title="")
                    st.plotly_chart(fig3, width="stretch")

            with b2:
                pareto = tdf["category"].value_counts().reset_index()
                pareto.columns = ["category","count"]
                pareto = pareto.sort_values("count", ascending=False)
                pareto["cum_pct"] = pareto["count"].cumsum() / pareto["count"].sum() * 100

                fig4 = make_subplots(specs=[[{"secondary_y": True}]])
                fig4.add_bar(x=pareto["category"], y=pareto["count"],
                    name="Count", marker_color="#0ea5a4", secondary_y=False)
                fig4.add_scatter(x=pareto["category"], y=pareto["cum_pct"],
                    mode="lines+markers", name="Cumulative %",
                    line=dict(color="#ef4444", width=2), secondary_y=True)
                fig4.update_layout(title="Category Pareto Analysis")
                fig4.update_yaxes(title_text="Count",        secondary_y=False)
                fig4.update_yaxes(title_text="Cumulative %", range=[0,105], secondary_y=True)
                st.plotly_chart(fig4, width="stretch")

            risk = tdf["urgency"].eq("High").mean() * 100
            fig5 = go.Figure(go.Indicator(
                mode="gauge+number", value=risk, number={"suffix":"%"},
                title={"text": "High-Urgency Load"},
                gauge={
                    "axis": {"range":[None,100]}, "bar":{"color":"#ef4444"},
                    "steps":[
                        {"range":[0,35],  "color":"#d1fae5"},
                        {"range":[35,65], "color":"#fef3c7"},
                        {"range":[65,100],"color":"#fee2e2"},
                    ],
                },
            ))
            fig5.update_layout(height=270, margin=dict(l=20,r=20,t=60,b=10))
            st.plotly_chart(fig5, width="stretch")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4 – RECORD EXPLORER
    # ════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Record Explorer")
        st.markdown('<p class="section-caption">Searchable, sortable records with full-message drilldown.</p>',
                    unsafe_allow_html=True)

        if filtered_df.empty:
            st.info("No records match current filters.")
        else:
            r1, r2, r3 = st.columns([1.2, 1, 0.9])
            with r1:
                sort_by = st.selectbox("Sort by", ["date","category","urgency"])
            with r2:
                order = st.radio("Order", ["Descending","Ascending"], horizontal=True)
            with r3:
                rows = st.number_input("Rows", min_value=5, max_value=200, value=15)

            disp = filtered_df.sort_values(sort_by, ascending=(order=="Ascending")).head(rows)
            tbl  = disp.copy()
            tbl["date"]          = tbl["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            tbl["email_preview"] = tbl["email"].astype(str).apply(
                lambda t: (t[:110]+"...") if len(t)>110 else t)

            st.dataframe(
                tbl[["date","category","urgency","email_preview"]],
                width="stretch", hide_index=True,
                column_config={
                    "date": "Date", "category": "Category",
                    "urgency": "Urgency", "email_preview": "Email Preview",
                },
            )

            st.divider()
            sel_idx = st.selectbox(
                "Inspect record",
                options=list(range(len(disp))),
                format_func=lambda i: (
                    f"{disp.iloc[i]['date'].strftime('%Y-%m-%d %H:%M')} | "
                    f"{disp.iloc[i]['category']} | {disp.iloc[i]['urgency']}"
                ),
            )
            rec = disp.iloc[sel_idx]
            d1, d2, d3 = st.columns(3)
            d1.metric("Date",     rec["date"].strftime("%Y-%m-%d %H:%M:%S"))
            d2.metric("Category", rec["category"])
            d3.metric("Urgency",  rec["urgency"])
            st.text_area("Full email", value=str(rec["email"]), height=220, disabled=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5 – BATCH CSV ANALYZER
    # ════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Batch CSV Analyzer")
        st.markdown('<p class="section-caption">Upload a CSV to classify multiple emails at once.</p>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"],
            help="CSV must contain at least one column with email text.")

        if uploaded:
            try:
                up_df = pd.read_csv(uploaded)
                st.markdown(f"**Preview** — {len(up_df)} rows · {len(up_df.columns)} columns")
                st.dataframe(up_df.head(5), hide_index=True, width="stretch")

                email_col = st.selectbox("Email-text column", up_df.columns.tolist())

                if st.button("▶️ Run Batch Classification", type="primary", width="stretch"):
                    emails  = up_df[email_col].fillna("").astype(str).tolist()
                    results = []
                    bar     = st.progress(0, text="Classifying…")

                    for i, txt in enumerate(emails):
                        try:
                            pred = predict_email(txt) if txt.strip() else {
                                "category":"unknown","urgency":"unknown",
                                "category_confidence":0.0,"urgency_confidence":0.0,"urgency_source":"none",
                            }
                        except Exception:
                            pred = {
                                "category":"error","urgency":"unknown",
                                "category_confidence":0.0,"urgency_confidence":0.0,"urgency_source":"none",
                            }
                        results.append({
                            "email":               txt,
                            "category":            pred["category"],
                            "urgency":             pred["urgency"],
                            "category_confidence": f"{pred['category_confidence']:.1%}",
                            "urgency_confidence":  f"{pred['urgency_confidence']:.1%}",
                            "urgency_source":      pred["urgency_source"],
                            "recommendation":      get_recommendation(pred["category"], pred["urgency"]),
                        })
                        bar.progress((i+1)/len(emails), text=f"Email {i+1} / {len(emails)}")

                    bar.empty()
                    st.session_state["batch_results"] = pd.DataFrame(results)
                    st.success(f"✅ {len(emails)} emails classified.")

            except Exception as e:
                st.error(f"Could not read file: {e}")

        batch = st.session_state.get("batch_results")
        if batch is not None and not batch.empty:
            st.markdown("### Prediction Results")

            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Total Classified", len(batch))
            b2.metric("High Urgency",     int((batch["urgency"]=="High").sum()))
            b3.metric("Complaints",       int((batch["category"]=="Complaint").sum()))
            b4.metric("Spam Detected",    int((batch["category"]=="Spam").sum()))

            bc1, bc2 = st.columns(2)
            with bc1:
                cc = batch["category"].value_counts().reset_index()
                cc.columns = ["category","count"]
                fig = px.bar(cc.sort_values("count", ascending=True),
                    x="count", y="category", orientation="h",
                    color="category", color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Category Distribution", text="count")
                fig.update_layout(showlegend=False, xaxis_title="Emails", yaxis_title="")
                st.plotly_chart(fig, width="stretch")

            with bc2:
                uc = batch["urgency"].value_counts().reindex(URGENCY_PRIORITY, fill_value=0)
                fig2 = go.Figure(data=[go.Pie(
                    labels=uc.index, values=uc.values, hole=0.55,
                    marker=dict(colors=["#ef4444","#f59e0b","#10b981"]),
                    textinfo="label+percent",
                )])
                fig2.update_layout(title="Urgency Composition")
                st.plotly_chart(fig2, width="stretch")

            st.dataframe(
                batch[["email","category","urgency","category_confidence",
                        "urgency_confidence","recommendation"]],
                hide_index=True, width="stretch",
                column_config={
                    "email": st.column_config.TextColumn("Email", width="large"),
                    "category":"Category","urgency":"Urgency",
                    "category_confidence":"Cat. Conf.","urgency_confidence":"Urg. Conf.",
                    "recommendation":"Recommendation",
                },
            )

            dl, sv = st.columns(2)
            with dl:
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=batch.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", width="stretch",
                )
            with sv:
                if st.button("💾 Save All to Dataset", width="stretch"):
                    n = 0
                    for _, row in batch.iterrows():
                        if row["category"] not in ("unknown","error"):
                            save_prediction({
                                "date": datetime.now(), "email": row["email"],
                                "category": row["category"], "urgency": row["urgency"],
                            }, DATA_FILE)
                            n += 1
                    st.success(f"{n} records saved.")
                    st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(f"Smart Email Intelligence Dashboard · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
