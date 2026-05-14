"""
Phase 4 — RAG (Retrieval Augmented Generation)
Run: python phase5_vector_db/rag.py
Needs: ANTHROPIC_API_KEY in .env
"""
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

DOCUMENTS = [
    "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes that process information by passing signals between them.",
    "Machine learning is a subset of AI where systems learn from data to improve their performance without being explicitly programmed for each task.",
    "Gradient descent is an optimization algorithm used to train neural networks. It adjusts weights by moving in the direction that reduces the loss function.",
    "Transformers are a neural network architecture that uses 'attention' to weigh the importance of different words in a sequence when making predictions.",
    "Overfitting happens when a model learns the training data too well, including noise, and performs poorly on new unseen data.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a language model that reads text in both directions to understand context better.",
    "Reinforcement learning is a type of machine learning where an agent learns by taking actions in an environment and receiving rewards or penalties.",
    "A loss function measures how wrong a model's predictions are. The goal of training is to minimize this value.",
    "Tokenization is the process of splitting text into smaller units (tokens) like words or subwords before feeding it into a language model.",
    "Fine-tuning is the process of taking a pre-trained model and continuing to train it on a smaller, specific dataset to adapt it to a particular task.",
]


def build_knowledge_base() -> chromadb.Collection:
    db = chromadb.EphemeralClient()
    collection = db.get_or_create_collection(
        name="ai_knowledge",
        embedding_function=embedding_fn,
    )
    collection.add(
        documents=DOCUMENTS,
        ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
    )
    print(f"[Knowledge base ready: {collection.count()} documents loaded]")
    return collection


def semantic_search(collection: chromadb.Collection, query: str, n_results: int = 3) -> list:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def demonstrate_search(collection: chromadb.Collection) -> None:
    queries = [
        "How do computers learn from data?",
        "Why does my model work on training but fail on new data?",
        "How does text get broken into pieces for AI?",
    ]
    for query in queries:
        print(f"\nQuery:  {query}")
        top_docs = semantic_search(collection, query, n_results=2)
        for i, doc in enumerate(top_docs, 1):
            print(f"  Result {i}: {doc[:100]}...")


def rag_answer(collection: chromadb.Collection, question: str) -> str:
    relevant_docs = semantic_search(collection, question, n_results=3)
    context = "\n\n".join(
        f"Document {i+1}: {doc}" for i, doc in enumerate(relevant_docs)
    )
    prompt = f"""Use ONLY the documents below to answer the question.
If the answer is not in the documents, say "I don't have that information."

--- DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Question: {question}
Answer:"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def rag_chatbot(collection: chromadb.Collection) -> None:
    print("\nRAG Chatbot — Ask me anything about AI concepts!")
    print("Type 'quit' to exit.")
    print("-" * 40)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        relevant_docs = semantic_search(collection, question, n_results=3)
        print(f"\n[Retrieved {len(relevant_docs)} relevant documents]")
        answer = rag_answer(collection, question)
        print(f"\nClaude: {answer}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4: RAG — Retrieval Augmented Generation")
    print("=" * 60)

    collection = build_knowledge_base()

    print("""
Choose an example:
  1 — Semantic search demo
  2 — Full RAG chatbot
""")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        demonstrate_search(collection)
    elif choice == "2":
        rag_chatbot(collection)
    else:
        print("Invalid choice. Running RAG chatbot by default.")
        rag_chatbot(collection)
