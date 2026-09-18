import os
import re
import json
from pypdf import PdfReader
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PDF_DIR = "data/hard_test_pdfs"
VECTORSTORE_PATH = "data/faiss_index"
OUTPUT_CSV = "data/evaluation_report.csv"


class FeedbackClassification(BaseModel):
    category: str = Field(description="One of: Excellent, Good, Need Improvements, Poor")
    confidence: float = Field(description="Confidence score between 0 and 1")
    rationale: str = Field(description="Brief plain-language explanation for this classification")


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def parse_feedback_and_truth(filename: str, raw_text: str):
    feedback_match = re.search(r"Feedback:\s*(.+)", raw_text, re.DOTALL)
    feedback_text = feedback_match.group(1).strip() if feedback_match else ""
    feedback_text = " ".join(feedback_text.split())

    # true label is baked into filename, e.g. hardtest_0_Need_Improvements.pdf
    label_part = filename.replace(".pdf", "").split("_", 2)[-1]
    true_label = label_part.replace("_", " ")

    return feedback_text, true_label


def build_chain():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    parser = PydanticOutputParser(pydantic_object=FeedbackClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert customer feedback analyst. Classify feedback into exactly one of these "
         "four categories:\n"
         "- Excellent: Strong satisfaction, delight, or advocacy; enthusiastic and affirming language.\n"
         "- Good: Generally satisfied with minor reservations; positive overall tone.\n"
         "- Need Improvements: Specific gaps flagged without strong dissatisfaction; constructive critique.\n"
         "- Poor: Frustration, disappointment, or churn intent; urgent negative sentiment.\n\n"
         "Judge INTENT and TONE, not just keywords. Mild dissatisfaction or reluctance to return is "
         "still 'Poor' if it signals churn risk, even without angry language.\n\n"
         "Here are similar past examples with verified labels, for reference:\n{examples}\n\n"
         "{format_instructions}"),
        ("human", "Classify this feedback:\n\n{feedback}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def format_examples(docs):
    return "\n".join(f"- \"{doc.page_content}\" -> {doc.metadata['label']}" for doc in docs)


def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_chain()

    rows = []
    for filename in sorted(os.listdir(PDF_DIR)):
        if not filename.endswith(".pdf"):
            continue

        raw_text = extract_text_from_pdf(os.path.join(PDF_DIR, filename))
        feedback_text, true_label = parse_feedback_and_truth(filename, raw_text)

        similar_docs = retriever.invoke(feedback_text)
        examples_str = format_examples(similar_docs)

        try:
            result = chain.invoke({"feedback": feedback_text, "examples": examples_str})
            predicted = result.category
            confidence = result.confidence
            rationale = result.rationale
        except Exception as e:
            predicted = "ERROR"
            confidence = 0.0
            rationale = str(e)

        correct = (predicted.strip().lower() == true_label.strip().lower())

        rows.append({
            "filename": filename,
            "feedback_text": feedback_text,
            "true_label": true_label,
            "predicted": predicted,
            "confidence": confidence,
            "correct": correct,
            "rationale": rationale,
        })

        status = "✅" if correct else "❌"
        print(f"{status} {filename} | true={true_label} | predicted={predicted}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    accuracy = df["correct"].mean() * 100
    print(f"\nAccuracy on hard test set: {accuracy:.1f}%  ({df['correct'].sum()}/{len(df)})")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    confusion = pd.crosstab(df["true_label"], df["predicted"])
    print(confusion)

    print(f"\nFull report saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()