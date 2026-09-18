import streamlit as st
import sys
import os
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from extract_text import extract_text_from_pdf, parse_fields
from classify import load_vectorstore, build_chain, format_examples

st.set_page_config(page_title="AI Customer Feedback Classifier", page_icon="📄", layout="centered")

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Classify Feedback", "Dashboard"])

CONFIDENCE_ALERT_THRESHOLD = 0.80


@st.cache_resource
def get_pipeline():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_chain()
    return retriever, chain


# ==========================================================
# PAGE 1: Classify a single feedback PDF
# ==========================================================
if page == "Classify Feedback":
    st.title("📄 AI-Powered Customer Feedback Classifier")
    st.write("Upload a customer feedback PDF (digital or scanned) to see it read, understood, and classified.")

    with st.expander("ℹ️ How this works"):
        st.markdown(
            "1. **Extract** — pypdf reads digital PDFs directly; if no text layer is found "
            "(e.g. a scanned form), it falls back to Tesseract OCR.\n"
            "2. **Retrieve** — similar past examples with verified labels are pulled from a FAISS "
            "vector store (RAG) to ground the classification.\n"
            "3. **Classify** — an LLM (Groq) judges intent and tone, not just keywords, and returns "
            "a category, confidence score, and plain-language rationale."
        )

    retriever, chain = get_pipeline()
    uploaded_file = st.file_uploader("Upload a feedback PDF", type=["pdf"])

    if uploaded_file is not None:
        temp_path = os.path.join("data", "temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading PDF..."):
            raw_text, method = extract_text_from_pdf(temp_path)
            fields = parse_fields(raw_text)

        st.success(f"Text extracted using: **{method.upper()}**")

        st.subheader("Extracted Fields")
        st.write(f"**Name:** {fields['name']}")
        st.write(f"**Date:** {fields['date']}")
        st.write(f"**Email:** {fields['email']}")
        st.write(f"**Feedback:** {fields['feedback_text']}")

        if fields['feedback_text']:
            with st.spinner("Classifying feedback..."):
                similar_docs = retriever.invoke(fields['feedback_text'])
                examples_str = format_examples(similar_docs)
                result = chain.invoke({
                    "feedback": fields['feedback_text'],
                    "examples": examples_str,
                })

            st.subheader("AI Classification")

            color_map = {
                "Excellent": "🟢",
                "Good": "🔵",
                "Need Improvements": "🟡",
                "Poor": "🔴",
            }
            emoji = color_map.get(result.category, "⚪")

            st.markdown(f"### {emoji} {result.category}")
            st.write(f"**Confidence:** {result.confidence:.2f}")
            st.write(f"**Rationale:** {result.rationale}")

            if result.category == "Poor" and result.confidence >= CONFIDENCE_ALERT_THRESHOLD:
                st.error("🚨 Auto-alert triggered: high-confidence churn risk detected!")

        os.remove(temp_path)


# ==========================================================
# PAGE 2: Dashboard — aggregate stats
# ==========================================================
elif page == "Dashboard":
    st.title("📊 Customer Feedback Dashboard")
    st.write("Aggregate insights from previously classified feedback.")

    DATA_FILE = "data/classified_feedback.json"

    if not os.path.exists(DATA_FILE):
        st.warning("No classified feedback found yet. Run `python src/classify.py` first.")
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if "category" in df.columns:
            df = df[df["category"].notna()]

        total = len(df)
        st.metric("Total Feedback Forms Processed", total)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Category Distribution")
            dist = df["category"].value_counts()
            st.bar_chart(dist)

        with col2:
            st.subheader("Average Confidence")
            avg_conf = df.groupby("category")["confidence"].mean().round(2)
            st.bar_chart(avg_conf)

        st.subheader("🚨 High-Risk Alerts (Poor, confidence ≥ 0.80)")
        alerts = df[(df["category"] == "Poor") & (df["confidence"] >= CONFIDENCE_ALERT_THRESHOLD)]
        st.write(f"{len(alerts)} response(s) would trigger an auto-ticket.")
        if len(alerts) > 0:
            st.dataframe(alerts[["filename", "feedback_text", "confidence"]])

        st.subheader("All Classified Feedback")
        st.dataframe(df[["filename", "feedback_text", "category", "confidence"]])