from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# A small set of VERIFIED, hand-labeled reference examples.
# This acts as our "ground truth" knowledge base for RAG retrieval.
REFERENCE_EXAMPLES = [
    
    # Excellent
    {"text": "Absolutely amazing service, I will tell everyone about this!", "label": "Excellent"},
    {"text": "The team went above and beyond, truly outstanding experience.", "label": "Excellent"},
    {"text": "I'm delighted with the quality, exceeded all expectations.", "label": "Excellent"},
    {"text": "Best customer service I've had in years, hats off to the team.", "label": "Excellent"},
    {"text": "Flawless from start to finish, this is how service should be.", "label": "Excellent"},
    {"text": "I'm genuinely impressed, this made my whole week better.", "label": "Excellent"},

    # Good
    {"text": "Overall good experience, just a small delay at checkout.", "label": "Good"},
    {"text": "Satisfied with the purchase, minor packaging issue though.", "label": "Good"},
    {"text": "Pretty solid service, nothing major to complain about.", "label": "Good"},
    {"text": "Happy with how things went, a couple of small hiccups but fine overall.", "label": "Good"},
    {"text": "Would use this again, the experience was mostly smooth.", "label": "Good"},
    {"text": "Decent value for money, does what it's supposed to do.", "label": "Good"},

    # Need Improvements
    {"text": "The app was a bit confusing and the wait felt long.", "label": "Need Improvements"},
    {"text": "Instructions were unclear, could use better guidance.", "label": "Need Improvements"},
    {"text": "Average service, there is room to improve response times.", "label": "Need Improvements"},
    {"text": "Not bad, but the process took more steps than it should.", "label": "Need Improvements"},
    {"text": "It works, but the interface really needs a redesign.", "label": "Need Improvements"},
    {"text": "Support was okay but took a while to actually resolve my issue.", "label": "Need Improvements"},

    # Poor
    {"text": "Terrible experience, I want my money back right now.", "label": "Poor"},
    {"text": "Extremely rude staff, I will never come back here.", "label": "Poor"},
    {"text": "Complete waste of time and money, very disappointed.", "label": "Poor"},
    {"text": "This is unacceptable, I'm cancelling my subscription today.", "label": "Poor"},
    {"text": "Nothing worked as promised, I feel completely misled.", "label": "Poor"},
    {"text": "Worst support I've dealt with, no one even responded.", "label": "Poor"},
]


VECTORSTORE_PATH = "data/faiss_index"


def build_vectorstore():
    # Local, free embedding model — no API key needed for this part
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [
        Document(page_content=ex["text"], metadata={"label": ex["label"]})
        for ex in REFERENCE_EXAMPLES
    ]

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"FAISS vector store built and saved to '{VECTORSTORE_PATH}' with {len(documents)} reference examples.")


if __name__ == "__main__":
    build_vectorstore()