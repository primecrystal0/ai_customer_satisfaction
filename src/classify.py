import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

VECTORSTORE_PATH = "data/faiss_index"
EXTRACTED_FILE = "data/extracted_feedback.json"
OUTPUT_FILE = "data/classified_feedback.json"


# ---- Step 1: Define the exact structured output we want (matches your slide) ----
class FeedbackClassification(BaseModel):
    category: str = Field(description="One of: Excellent, Good, Need Improvements, Poor")
    confidence: float = Field(description="Confidence score between 0 and 1")
    rationale: str = Field(description="Brief plain-language explanation for this classification")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)


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
         "Judge INTENT and TONE, not just keywords. For example, 'the wait was a bit long' is "
         "'Need Improvements', not 'Poor', because it's mild constructive feedback, not anger.\n\n"
         "Here are similar past examples with verified labels, for reference:\n{examples}\n\n"
         "{format_instructions}"),
        ("human", "Classify this feedback:\n\n{feedback}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def format_examples(docs):
    return "\n".join(f"- \"{doc.page_content}\" -> {doc.metadata['label']}" for doc in docs)


def main():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_chain()

    with open(EXTRACTED_FILE, "r", encoding="utf-8") as f:
        forms = json.load(f)

    results = []
    for form in forms:
        feedback_text = form["feedback_text"]
        if not feedback_text:
            continue

        similar_docs = retriever.invoke(feedback_text)
        examples_str = format_examples(similar_docs)

        try:
            classification = chain.invoke({
                "feedback": feedback_text,
                "examples": examples_str,
            })
            result = {
                "filename": form["filename"],
                "name": form["name"],
                "feedback_text": feedback_text,
                "category": classification.category,
                "confidence": classification.confidence,
                "rationale": classification.rationale,
            }
        except Exception as e:
            result = {
                "filename": form["filename"],
                "feedback_text": feedback_text,
                "error": str(e),
            }

        results.append(result)
        print(f"{form['filename']} -> {result.get('category', 'ERROR')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} classifications to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()