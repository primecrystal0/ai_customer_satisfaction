# AI-Powered Customer Satisfaction from PDF Feedback

An end-to-end AI pipeline that reads customer feedback PDFs, classifies each response into one of four satisfaction categories, and surfaces actionable insights — built for the Cognizant NPN Hackathon.

## Problem

Thousands of customer feedback PDFs go unread. Manual review is slow, inconsistent, and misses critical signals like churn risk hidden in casual language.

## Solution

This pipeline reads, understands, and classifies feedback in seconds using an LLM grounded with Retrieval-Augmented Generation (RAG), producing a category, confidence score, and plain-language rationale for every response — fully auditable.

## Architecture

| Layer | Tech Used |
|---|---|
| Ingestion | `reportlab` (synthetic test PDFs), `faker` (realistic sample data) |
| Text Extraction | `pypdf` + regex field parsing |
| Embeddings / Vector Store | `sentence-transformers` (local, free) + FAISS |
| LLM Classification | Groq (`openai/gpt-oss-20b`) via `langchain-groq` |
| Structured Output | Pydantic (category, confidence, rationale) |
| Dashboard | `pandas` — category distribution, avg confidence, auto-alerts |

## The 4 Categories

- **Excellent** — Strong satisfaction, delight, or advocacy
- **Good** — Generally satisfied, minor reservations
- **Need Improvements** — Constructive critique, no strong dissatisfaction
- **Poor** — Frustration, disappointment, or churn intent

## Results

Evaluated on a deliberately hard, unseen test set (ambiguous phrasing, mild dissatisfaction, mixed tone — not just obvious keyword matches):

- **100% accuracy** (10/10) on the hard test set
- **100% accuracy** (20/20) on the base test set
- Full confusion matrix and per-response rationale available in `data/evaluation_report.csv`

This validates the "intent over keywords" design goal — e.g. *"It's alright I guess, wouldn't go out of my way to use it again"* is correctly classified as **Poor** (churn signal) despite having no negative keywords.

## How It Works

1. `src/generate_pdfs.py` — generates synthetic feedback PDFs for testing
2. `src/extract_text.py` — extracts and parses structured fields from PDFs
3. `src/build_vectorstore.py` — builds a FAISS vector store of verified labeled examples for RAG grounding
4. `src/classify.py` — classifies feedback using Groq LLM + RAG retrieval, outputs category/confidence/rationale
5. `src/generate_hard_testset.py` + `src/evaluate.py` — generates a harder unseen test set and reports accuracy + confusion matrix
6. `src/dashboard.py` — summarizes results into category distribution, confidence stats, and auto-alerts for high-risk (Poor) responses

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Add your Groq API key to a `.env` file:

## Run the Web App

```bash
streamlit run app.py
```

Upload any feedback PDF (digital or scanned) to see it extracted and classified live. Switch to the **Dashboard** tab in the sidebar to view aggregate statistics across previously classified feedback.


## Run the Pipeline

```bash
python src/generate_pdfs.py          # create sample PDFs
python src/extract_text.py           # extract text + fields
python src/build_vectorstore.py      # build RAG knowledge base
python src/classify.py               # classify all feedback
python src/dashboard.py              # view summary dashboard
```

To evaluate accuracy on a harder test set:
```bash
python src/generate_hard_testset.py
python src/evaluate.py
```

## Future Work

- Handwriting-specific OCR fine-tuning (current OCR is tested on printed/scanned text; real handwritten forms are a documented next step)
- Live BI dashboard (Power BI/Looker) integration for production-scale deployment
- CRM/ticketing auto-alert integration (ServiceNow/Salesforce)

## Author

Rajeev — Cognizant NPN Hackathon, September 2026
