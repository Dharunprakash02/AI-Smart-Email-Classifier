import os
import re
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, DistilBertForSequenceClassification


st.set_page_config(
    page_title="Smart Email Intelligence Dashboard",
    page_icon="mailbox",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_FILE = "email_predictions.csv"
CATEGORY_LABELS = ["complaint", "request", "feedback", "spam", "inquiry", "other"]
URGENCY_LABELS = ["low", "medium", "high"]
URGENCY_PRIORITY = ["high", "medium", "low"]
URGENT_KEYWORDS = [
    "urgent",
    "asap",
    "immediately",
    "not working",
    "server down",
    "critical",
    "error",
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "your",
    "please",
    "would",
    "there",
    "their",
    "about",
    "they",
    "them",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "into",
    "just",
    "need",
    "hello",
    "regards",
    "thanks",
    "thank",
    "subject",
    "email",
    "message",
    "team",
    "dear",
}


def render_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink-900: #e2e8f0;
                --ink-700: #cbd5e1;
                --ink-500: #94a3b8;
                --sky-500: #0ea5a4;
                --sky-300: #7dd3fc;
                --surface-100: #111827;
                --surface-200: #1f2937;
                --success-500: #10b981;
                --warn-500: #f59e0b;
                --danger-500: #ef4444;
            }

            .stApp {
                background:
                    radial-gradient(900px 400px at 10% -10%, #0b1220 0%, transparent 55%),
                    radial-gradient(700px 350px at 95% 0%, #0a1320 0%, transparent 50%),
                    linear-gradient(180deg, #030712 0%, #000000 100%);
                color: #e5e7eb;
            }

            .stMarkdown,
            .stMarkdown p,
            .stMarkdown li,
            .st-emotion-cache-10trblm,
            [data-testid="stSidebar"] * {
                color: #e5e7eb;
            }

            .hero {
                background: linear-gradient(125deg, #0f172a 0%, #134e4a 100%);
                border-radius: 16px;
                padding: 1.2rem 1.4rem;
                border: 1px solid rgba(255, 255, 255, 0.16);
                margin-bottom: 1rem;
            }

            .hero h1 {
                margin: 0;
                color: #ffffff;
                font-size: 1.9rem;
                letter-spacing: 0.2px;
                font-weight: 700;
            }

            .hero p {
                margin: 0.4rem 0 0;
                color: #dbeafe;
                font-size: 0.95rem;
            }

            [data-testid="stMetric"] {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 0.4rem 0.6rem;
                box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.45rem;
                padding-bottom: 0.2rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: #1f2937;
                border-radius: 10px;
                color: #d1d5db;
                font-weight: 600;
                padding: 0.45rem 0.9rem;
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(130deg, #0e7490 0%, #0284c7 100%);
                color: #ffffff;
            }

            .section-caption {
                color: var(--ink-500);
                font-size: 0.92rem;
                margin-top: -0.2rem;
                margin-bottom: 0.9rem;
            }

            .result-box {
                border: 1px solid #334155;
                background: #111827;
                border-radius: 12px;
                padding: 0.85rem 1rem;
                margin-top: 0.8rem;
                color: #e2e8f0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_data_file(path: str) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=["date", "email", "category", "urgency"]).to_csv(path, index=False)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    ensure_data_file(path)
    df = pd.read_csv(path)

    expected_cols = ["date", "email", "category", "urgency"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_cols]
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def save_prediction(record: dict, path: str) -> None:
    ensure_data_file(path)
    try:
        current_df = pd.read_csv(path)
    except Exception:
        current_df = pd.DataFrame(columns=["date", "email", "category", "urgency"])

    updated_df = pd.concat([current_df, pd.DataFrame([record])], ignore_index=True)
    updated_df.to_csv(path, index=False)
    load_data.clear()


@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer", local_files_only=True)
    category_model = DistilBertForSequenceClassification.from_pretrained(
        "models/category_model",
        local_files_only=True,
    )
    urgency_model = DistilBertForSequenceClassification.from_pretrained(
        "models/urgency_model",
        local_files_only=True,
    )

    category_model.eval()
    urgency_model.eval()
    return tokenizer, category_model, urgency_model


def rule_based_urgency(text: str):
    lowered = text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in lowered:
            return "high"
    return None


def predict_email(text: str) -> dict:
    tokenizer, category_model, urgency_model = load_models()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.inference_mode():
        category_logits = category_model(**inputs).logits
        category_probs = torch.softmax(category_logits, dim=-1).squeeze(0)
        cat_idx = int(torch.argmax(category_probs).item())
        category_distribution = {
            label: float(category_probs[idx].item()) for idx, label in enumerate(CATEGORY_LABELS)
        }

        urgency_rule = rule_based_urgency(text)
        if urgency_rule:
            urgency_label = urgency_rule
            urgency_confidence = 1.0
            urgency_source = "rule"
            urgency_distribution = {"low": 0.0, "medium": 0.0, "high": 1.0}
        else:
            urgency_logits = urgency_model(**inputs).logits
            urgency_probs = torch.softmax(urgency_logits, dim=-1).squeeze(0)
            urg_idx = int(torch.argmax(urgency_probs).item())
            urgency_label = URGENCY_LABELS[urg_idx]
            urgency_confidence = float(urgency_probs[urg_idx].item())
            urgency_source = "model"
            urgency_distribution = {
                label: float(urgency_probs[idx].item()) for idx, label in enumerate(URGENCY_LABELS)
            }

    return {
        "category": CATEGORY_LABELS[cat_idx],
        "urgency": urgency_label,
        "category_confidence": float(category_probs[cat_idx].item()),
        "urgency_confidence": urgency_confidence,
        "urgency_source": urgency_source,
        "category_distribution": category_distribution,
        "urgency_distribution": urgency_distribution,
    }


def get_recommendation(category: str, urgency: str) -> str:
    if urgency == "high":
        return "Escalate to operations and assign immediate owner."
    if category == "complaint":
        return "Open support ticket and respond with resolution timeline."
    if category == "request":
        return "Assign to service desk queue and confirm expected turnaround."
    if category == "spam":
        return "Flag as spam and exclude from action queues."
    return "Route to standard triage for follow-up."


def build_report_pdf(report: dict):
    try:
        from fpdf import FPDF
    except Exception:
        return None, "PDF export requires fpdf2. Install with: pip install fpdf2"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def safe_multiline(text: str, line_height: int = 6) -> None:
        # Keep cursor at left margin after each line so width stays valid across writes.
        pdf.multi_cell(0, line_height, str(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Email Classification Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Generated: {report['timestamp']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    safe_multiline(f"Category: {report['category'].title()} ({report['category_confidence']:.1%})")
    safe_multiline(f"Urgency: {report['urgency'].upper()} ({report['urgency_confidence']:.1%})")
    safe_multiline(f"Urgency source: {report['urgency_source']}")
    safe_multiline(f"Recommended action: {report['recommendation']}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Category Probabilities", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for label, value in sorted(report["category_distribution"].items(), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 6, f"- {label.title()}: {value:.1%}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Urgency Probabilities", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for label, value in sorted(report["urgency_distribution"].items(), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 6, f"- {label.upper()}: {value:.1%}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Email Content", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    preview = report["email"][:1800]
    if len(report["email"]) > 1800:
        preview += "..."
    safe_multiline(preview, line_height=5)

    pdf_bytes = bytes(pdf.output(dest="S"))
    return pdf_bytes, None


def render_prediction_report(report: dict) -> None:
    st.markdown("### Analysis Report")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Category", report["category"].title())
    m2.metric("Predicted Urgency", report["urgency"].upper())
    m3.metric("Urgency Source", report["urgency_source"].title())

    s1, s2 = st.columns(2)
    s1.metric("Category Confidence", f"{report['category_confidence']:.1%}")
    s2.metric("Urgency Confidence", f"{report['urgency_confidence']:.1%}")

    st.info(f"Recommended action: {report['recommendation']}")

    category_df = (
        pd.DataFrame(
            [{"category": key, "probability": value} for key, value in report["category_distribution"].items()]
        )
        .sort_values("probability", ascending=True)
        .reset_index(drop=True)
    )

    urgency_df = pd.DataFrame(
        [{"urgency": key, "probability": value} for key, value in report["urgency_distribution"].items()]
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_cat = px.bar(
            category_df,
            x="probability",
            y="category",
            orientation="h",
            title="Category Probability Breakdown",
            text=category_df["probability"].map(lambda v: f"{v:.1%}"),
            color="probability",
            color_continuous_scale="Teal",
        )
        fig_cat.update_layout(showlegend=False, xaxis_title="Probability", yaxis_title="")
        st.plotly_chart(fig_cat, width="stretch")

    with c2:
        urgency_color_map = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}
        fig_urg = go.Figure(
            data=[
                go.Pie(
                    labels=urgency_df["urgency"],
                    values=urgency_df["probability"],
                    hole=0.55,
                    marker=dict(colors=[urgency_color_map.get(item, "#64748b") for item in urgency_df["urgency"]]),
                    textinfo="label+percent",
                )
            ]
        )
        fig_urg.update_layout(title="Urgency Probability Breakdown")
        st.plotly_chart(fig_urg, width="stretch")

    high_risk_value = report["urgency_distribution"].get("high", 0.0) * 100
    fig_risk = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=high_risk_value,
            number={"suffix": "%"},
            title={"text": "High Urgency Risk"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "#ef4444"},
                "steps": [
                    {"range": [0, 35], "color": "#d1fae5"},
                    {"range": [35, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig_risk.update_layout(height=280)
    st.plotly_chart(fig_risk, width="stretch")

    pdf_bytes, pdf_error = build_report_pdf(report)
    if pdf_bytes:
        st.download_button(
            "Download PDF Report",
            data=BytesIO(pdf_bytes),
            file_name=f"email_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    else:
        st.warning(pdf_error)


def apply_filters(df: pd.DataFrame, date_range, selected_categories, selected_urgencies, search_term: str):
    if df.empty:
        return df

    filtered = df.copy()

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["date"].dt.date >= start_date) & (filtered["date"].dt.date <= end_date)
        ]

    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    else:
        filtered = filtered.iloc[0:0]

    if selected_urgencies:
        filtered = filtered[filtered["urgency"].isin(selected_urgencies)]
    else:
        filtered = filtered.iloc[0:0]

    if search_term.strip():
        filtered = filtered[
            filtered["email"].str.contains(search_term.strip(), case=False, na=False)
        ]

    return filtered


def extract_top_keywords(email_series: pd.Series, top_n: int = 12) -> pd.DataFrame:
    words = []
    for text in email_series.fillna(""):
        tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        words.extend(token for token in tokens if token not in STOPWORDS)

    if not words:
        return pd.DataFrame(columns=["keyword", "count"])

    keyword_df = pd.Series(words).value_counts().head(top_n).reset_index()
    keyword_df.columns = ["keyword", "count"]
    return keyword_df


def main() -> None:
    render_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Smart Email Intelligence Dashboard</h1>
            <p>Professional classification, urgency detection, and operational analytics in one streamlined control center.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data(DATA_FILE)

    with st.sidebar:
        st.header("Filters")

        if df.empty:
            st.info("No records found. Submit a new email in the Analyzer tab to start building analytics.")
            filtered_df = df
        else:
            quick_filter = st.selectbox(
                "Quick filter",
                ["Custom", "High urgency only", "Complaints only", "Last 7 days"],
                index=0,
            )

            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            default_start = max(min_date, max_date - timedelta(days=30))

            date_range = st.date_input(
                "Date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            category_options = sorted(set(CATEGORY_LABELS) | set(df["category"].dropna().unique()))
            category_choice = st.selectbox(
                "Category",
                options=(["All"] + category_options) if category_options else ["All"],
                index=0,
            )
            if category_choice == "All":
                selected_categories = category_options
            else:
                selected_categories = [category_choice]

            available_urgencies = [u for u in URGENCY_PRIORITY if u in set(df["urgency"].dropna().unique())]
            urgency_options = URGENCY_PRIORITY
            urgency_choice = st.selectbox(
                "Urgency",
                options=["All"] + urgency_options,
                index=0,
            )
            if urgency_choice == "All":
                selected_urgencies = available_urgencies
            else:
                selected_urgencies = [urgency_choice]

            search_term = st.text_input("Search email text", placeholder="Type keywords...")

            filtered_df = apply_filters(
                df,
                date_range,
                selected_categories,
                selected_urgencies,
                search_term,
            )

            if quick_filter == "High urgency only":
                filtered_df = filtered_df[filtered_df["urgency"] == "high"]
            elif quick_filter == "Complaints only":
                filtered_df = filtered_df[filtered_df["category"] == "complaint"]
            elif quick_filter == "Last 7 days":
                cutoff = datetime.now().date() - timedelta(days=7)
                filtered_df = filtered_df[filtered_df["date"].dt.date >= cutoff]

            st.divider()
            st.metric("Filtered records", len(filtered_df))
            st.metric("Total records", len(df))

            if not filtered_df.empty:
                export_csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export filtered CSV",
                    data=export_csv,
                    file_name=f"filtered_emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width="stretch",
                )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Analyzer", "Executive Dashboard", "Deep Analytics", "Record Explorer", "Batch CSV Analyzer"]
    )

    with tab1:
        st.subheader("Classify New Email")
        st.markdown(
            '<p class="section-caption">Submit an email, run AI classification, and log the result to your dataset.</p>',
            unsafe_allow_html=True,
        )

        col_form, col_quick = st.columns([3, 1.2])

        with col_form:
            with st.form("email_analysis_form", clear_on_submit=False):
                email_text = st.text_area(
                    "Email content",
                    height=220,
                    placeholder="Paste the customer or internal email text here...",
                )
                submitted = st.form_submit_button(
                    "Analyze Email",
                    width="stretch",
                    type="primary",
                )

        with col_quick:
            word_count = len(email_text.split()) if isinstance(email_text, str) else 0
            char_count = len(email_text) if isinstance(email_text, str) else 0
            reading_time = max(1, round(word_count / 200)) if word_count else 0

            st.metric("Word count", word_count)
            st.metric("Characters", char_count)
            st.metric("Read time", f"{reading_time} min" if reading_time else "0 min")

        if submitted:
            if not email_text.strip():
                st.warning("Enter email text before running analysis.")
            else:
                try:
                    with st.status("Running classification pipeline...", expanded=True) as status:
                        status.write("Loading tokenizer and models. This happens once and stays cached.")
                        result = predict_email(email_text)
                        status.write("Writing prediction to dataset.")
                        report_payload = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "email": email_text,
                            "category": result["category"],
                            "urgency": result["urgency"],
                            "category_confidence": result["category_confidence"],
                            "urgency_confidence": result["urgency_confidence"],
                            "urgency_source": result["urgency_source"],
                            "category_distribution": result["category_distribution"],
                            "urgency_distribution": result["urgency_distribution"],
                            "recommendation": get_recommendation(result["category"], result["urgency"]),
                        }

                        save_prediction(
                            {
                                "date": datetime.now(),
                                "email": email_text,
                                "category": result["category"],
                                "urgency": result["urgency"],
                            },
                            DATA_FILE,
                        )
                        st.session_state["latest_report"] = report_payload
                        status.update(label="Email analyzed successfully", state="complete")

                    st.success("Prediction completed and saved.")

                    st.markdown(
                        """
                        <div class="result-box">
                            <strong>Result summary</strong><br>
                            Detailed report generated below with category and urgency charts.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.error(f"Unable to process email right now: {exc}")

        latest_report = st.session_state.get("latest_report")
        if latest_report:
            render_prediction_report(latest_report)
            with st.expander("Submitted email content"):
                st.write(latest_report.get("email", ""))

    with tab2:
        st.subheader("Executive Dashboard")
        st.markdown(
            '<p class="section-caption">A clean operating view of volume, urgency pressure, and category composition.</p>',
        )

        if filtered_df.empty:
            st.info("No records match the selected filters.")
        else:
            total = len(filtered_df)
            high_urgency_count = int((filtered_df["urgency"] == "high").sum())
            complaint_count = int((filtered_df["category"] == "complaint").sum())
            complaint_rate = (complaint_count / total) * 100 if total else 0

            unique_days = max(1, filtered_df["date"].dt.date.nunique())
            daily_average = total / unique_days

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total emails", total)
            m2.metric("High urgency", high_urgency_count, delta=f"{(high_urgency_count / total):.1%}")
            m3.metric("Complaint share", f"{complaint_rate:.1f}%")
            m4.metric("Avg emails per day", f"{daily_average:.1f}")

            c1, c2 = st.columns([1.35, 1])

            with c1:
                category_counts = filtered_df["category"].value_counts().reset_index()
                category_counts.columns = ["category", "count"]
                category_counts = category_counts.sort_values("count", ascending=True)

                fig_category = px.bar(
                    category_counts,
                    x="count",
                    y="category",
                    orientation="h",
                    text="count",
                    color="category",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Category distribution",
                )
                fig_category.update_layout(showlegend=False, xaxis_title="Emails", yaxis_title="")
                st.plotly_chart(fig_category, width="stretch")

            with c2:
                urgency_counts = filtered_df["urgency"].value_counts().reindex(URGENCY_PRIORITY, fill_value=0)
                fig_urgency = go.Figure(
                    data=[
                        go.Pie(
                            labels=urgency_counts.index,
                            values=urgency_counts.values,
                            hole=0.62,
                            marker=dict(colors=["#ef4444", "#f59e0b", "#10b981"]),
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_urgency.update_layout(title="Urgency composition", margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_urgency, width="stretch")

            st.divider()

            trend_df = (
                filtered_df.assign(day=filtered_df["date"].dt.date)
                .groupby("day")
                .size()
                .reset_index(name="count")
            )
            trend_df["day"] = pd.to_datetime(trend_df["day"])
            trend_df["moving_avg_7"] = trend_df["count"].rolling(window=7, min_periods=1).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_df["day"],
                    y=trend_df["count"],
                    mode="lines+markers",
                    name="Daily volume",
                    line=dict(color="#0f766e", width=2.2),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_df["day"],
                    y=trend_df["moving_avg_7"],
                    mode="lines",
                    name="7-day moving average",
                    line=dict(color="#0284c7", width=2, dash="dot"),
                )
            )
            fig_trend.update_layout(
                title="Email volume trend",
                xaxis_title="Date",
                yaxis_title="Count",
                hovermode="x unified",
            )
            st.plotly_chart(fig_trend, width="stretch")

            crosstab = pd.crosstab(filtered_df["category"], filtered_df["urgency"])
            crosstab = crosstab.reindex(columns=URGENCY_PRIORITY, fill_value=0)
            fig_heatmap = px.imshow(
                crosstab,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Teal",
                title="Category vs urgency intensity",
                labels={"x": "Urgency", "y": "Category", "color": "Emails"},
            )
            st.plotly_chart(fig_heatmap, width="stretch")

    with tab3:
        st.subheader("Deep Analytics")
        st.markdown(
            '<p class="section-caption">Statistical behavior patterns and operational risk indicators.</p>',
            unsafe_allow_html=True,
        )

        if filtered_df.empty:
            st.info("No records available for deep analytics.")
        else:
            temp_df = filtered_df.copy()
            temp_df["weekday"] = temp_df["date"].dt.day_name()
            temp_df["hour"] = temp_df["date"].dt.hour

            weekday_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            activity_pivot = pd.pivot_table(
                temp_df,
                index="weekday",
                columns="hour",
                values="email",
                aggfunc="count",
                fill_value=0,
            ).reindex(weekday_order)

            a1, a2 = st.columns(2)

            with a1:
                fig_activity = px.imshow(
                    activity_pivot,
                    aspect="auto",
                    color_continuous_scale="YlGnBu",
                    title="Weekday-hour traffic heatmap",
                    labels={"x": "Hour of day", "y": "Weekday", "color": "Emails"},
                )
                st.plotly_chart(fig_activity, width="stretch")

            with a2:
                urgency_daily = (
                    temp_df.assign(day=temp_df["date"].dt.date)
                    .groupby(["day", "urgency"])
                    .size()
                    .reset_index(name="count")
                )

                fig_urgency_area = px.area(
                    urgency_daily,
                    x="day",
                    y="count",
                    color="urgency",
                    title="Urgency load over time",
                    color_discrete_map={"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"},
                )
                fig_urgency_area.update_layout(xaxis_title="Date", yaxis_title="Email count")
                st.plotly_chart(fig_urgency_area, width="stretch")

            b1, b2 = st.columns(2)

            with b1:
                keyword_df = extract_top_keywords(temp_df["email"], top_n=12)
                if keyword_df.empty:
                    st.info("Not enough text data for keyword extraction.")
                else:
                    fig_keywords = px.bar(
                        keyword_df.sort_values("count", ascending=True),
                        x="count",
                        y="keyword",
                        orientation="h",
                        color="count",
                        color_continuous_scale="Blues",
                        title="Most frequent terms",
                    )
                    fig_keywords.update_layout(showlegend=False, xaxis_title="Mentions", yaxis_title="")
                    st.plotly_chart(fig_keywords, width="stretch")

            with b2:
                pareto_df = temp_df["category"].value_counts().reset_index()
                pareto_df.columns = ["category", "count"]
                pareto_df = pareto_df.sort_values("count", ascending=False)
                pareto_df["cum_pct"] = pareto_df["count"].cumsum() / pareto_df["count"].sum() * 100

                fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pareto.add_bar(
                    x=pareto_df["category"],
                    y=pareto_df["count"],
                    name="Count",
                    marker_color="#0ea5a4",
                    secondary_y=False,
                )
                fig_pareto.add_scatter(
                    x=pareto_df["category"],
                    y=pareto_df["cum_pct"],
                    mode="lines+markers",
                    name="Cumulative %",
                    line=dict(color="#ef4444", width=2),
                    secondary_y=True,
                )
                fig_pareto.update_layout(title="Category Pareto analysis")
                fig_pareto.update_yaxes(title_text="Count", secondary_y=False)
                fig_pareto.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
                st.plotly_chart(fig_pareto, width="stretch")

            risk_ratio = (temp_df["urgency"].eq("high").mean() * 100) if not temp_df.empty else 0
            fig_risk = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_ratio,
                    number={"suffix": "%"},
                    title={"text": "High-Urgency Load"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "#ef4444"},
                        "steps": [
                            {"range": [0, 35], "color": "#d1fae5"},
                            {"range": [35, 65], "color": "#fef3c7"},
                            {"range": [65, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            fig_risk.update_layout(height=270, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig_risk, width="stretch")

    with tab4:
        st.subheader("Record Explorer")
        st.markdown(
            '<p class="section-caption">Searchable and sortable email records with full-message drilldown.</p>',
            unsafe_allow_html=True,
        )

        if filtered_df.empty:
            st.info("No records available under the current filters.")
        else:
            r_col1, r_col2, r_col3 = st.columns([1.2, 1, 0.9])

            with r_col1:
                sort_by = st.selectbox("Sort by", ["date", "category", "urgency"], index=0)

            with r_col2:
                order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

            with r_col3:
                rows_to_show = st.number_input("Rows", min_value=5, max_value=200, value=15)

            ascending = order == "Ascending"
            display_df = filtered_df.sort_values(by=sort_by, ascending=ascending).head(rows_to_show)

            table_df = display_df.copy()
            table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            table_df["email_preview"] = table_df["email"].astype(str).apply(
                lambda text: (text[:110] + "...") if len(text) > 110 else text
            )

            st.dataframe(
                table_df[["date", "category", "urgency", "email_preview"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "date": "Date",
                    "category": "Category",
                    "urgency": "Urgency",
                    "email_preview": "Email preview",
                },
            )

            st.divider()
            selected_position = st.selectbox(
                "Inspect record",
                options=list(range(len(display_df))),
                format_func=lambda i: (
                    f"{display_df.iloc[i]['date'].strftime('%Y-%m-%d %H:%M')} | "
                    f"{display_df.iloc[i]['category'].title()} | "
                    f"{display_df.iloc[i]['urgency'].upper()}"
                ),
            )

            selected_record = display_df.iloc[selected_position]

            d1, d2, d3 = st.columns(3)
            d1.metric("Date", selected_record["date"].strftime("%Y-%m-%d %H:%M:%S"))
            d2.metric("Category", str(selected_record["category"]).title())
            d3.metric("Urgency", str(selected_record["urgency"]).upper())

            st.text_area("Full email", value=str(selected_record["email"]), height=220, disabled=True)

    with tab5:
        st.subheader("Batch CSV Analyzer")
        st.markdown(
            '<p class="section-caption">Upload a CSV file to classify multiple emails at once and download or save the results.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Your CSV must contain at least one column with email text.",
        )

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Preview** — {len(upload_df)} rows, {len(upload_df.columns)} columns")
                st.dataframe(upload_df.head(5), hide_index=True, use_container_width=True)

                email_col = st.selectbox(
                    "Select the column containing email text",
                    options=upload_df.columns.tolist(),
                )

                if st.button("Run Batch Classification", type="primary", use_container_width=True):
                    emails = upload_df[email_col].fillna("").astype(str).tolist()
                    results = []
                    progress_bar = st.progress(0, text="Classifying emails...")
                    total_emails = len(emails)

                    for i, email_text in enumerate(emails):
                        try:
                            if email_text.strip():
                                pred = predict_email(email_text)
                            else:
                                pred = {
                                    "category": "unknown",
                                    "urgency": "unknown",
                                    "category_confidence": 0.0,
                                    "urgency_confidence": 0.0,
                                    "urgency_source": "none",
                                }
                        except Exception:
                            pred = {
                                "category": "error",
                                "urgency": "unknown",
                                "category_confidence": 0.0,
                                "urgency_confidence": 0.0,
                                "urgency_source": "none",
                            }

                        results.append(
                            {
                                "email": email_text,
                                "category": pred["category"],
                                "urgency": pred["urgency"],
                                "category_confidence": f"{pred['category_confidence']:.1%}",
                                "urgency_confidence": f"{pred['urgency_confidence']:.1%}",
                                "urgency_source": pred["urgency_source"],
                                "recommendation": get_recommendation(
                                    pred["category"], pred["urgency"]
                                ),
                            }
                        )
                        progress_bar.progress(
                            (i + 1) / total_emails,
                            text=f"Classifying email {i + 1} of {total_emails}...",
                        )

                    progress_bar.empty()
                    st.session_state["batch_results"] = pd.DataFrame(results)
                    st.success(f"Batch classification complete — {total_emails} emails processed.")

            except Exception as exc:
                st.error(f"Failed to read the uploaded file: {exc}")

        batch_results = st.session_state.get("batch_results")
        if batch_results is not None and not batch_results.empty:
            st.markdown("### Prediction Results")

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total classified", len(batch_results))
            b2.metric("High urgency", int((batch_results["urgency"] == "high").sum()))
            b3.metric("Complaints", int((batch_results["category"] == "complaint").sum()))
            b4.metric("Spam detected", int((batch_results["category"] == "spam").sum()))

            bc1, bc2 = st.columns(2)
            with bc1:
                cat_counts = batch_results["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig_b_cat = px.bar(
                    cat_counts.sort_values("count", ascending=True),
                    x="count",
                    y="category",
                    orientation="h",
                    color="category",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title="Category distribution",
                    text="count",
                )
                fig_b_cat.update_layout(showlegend=False, xaxis_title="Emails", yaxis_title="")
                st.plotly_chart(fig_b_cat, width="stretch")

            with bc2:
                valid_urgencies = [u for u in URGENCY_PRIORITY if u in batch_results["urgency"].values]
                urg_counts = (
                    batch_results["urgency"]
                    .value_counts()
                    .reindex(URGENCY_PRIORITY, fill_value=0)
                )
                fig_b_urg = go.Figure(
                    data=[
                        go.Pie(
                            labels=urg_counts.index,
                            values=urg_counts.values,
                            hole=0.55,
                            marker=dict(colors=["#ef4444", "#f59e0b", "#10b981"]),
                            textinfo="label+percent",
                        )
                    ]
                )
                fig_b_urg.update_layout(title="Urgency composition")
                st.plotly_chart(fig_b_urg, width="stretch")

            st.dataframe(
                batch_results[
                    [
                        "email",
                        "category",
                        "urgency",
                        "category_confidence",
                        "urgency_confidence",
                        "recommendation",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "email": st.column_config.TextColumn("Email", width="large"),
                    "category": "Category",
                    "urgency": "Urgency",
                    "category_confidence": "Cat. Confidence",
                    "urgency_confidence": "Urg. Confidence",
                    "recommendation": "Recommendation",
                },
            )

            dl_col, save_col = st.columns(2)
            with dl_col:
                csv_out = batch_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_out,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with save_col:
                if st.button("Save All to Dataset", use_container_width=True):
                    saved_count = 0
                    for _, row in batch_results.iterrows():
                        if row["category"] not in ("unknown", "error"):
                            save_prediction(
                                {
                                    "date": datetime.now(),
                                    "email": row["email"],
                                    "category": row["category"],
                                    "urgency": row["urgency"],
                                },
                                DATA_FILE,
                            )
                            saved_count += 1
                    st.success(f"{saved_count} records saved to dataset.")
                    st.rerun()

    st.divider()
    st.caption(
        f"Smart Email Intelligence Dashboard | Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
