"""
Phase 5 — Full Chatbot Web App
Run: streamlit run phase6_chatbot/app.py
Needs: ANTHROPIC_API_KEY in .env
"""
import datetime
import pathlib
import anthropic
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

MODEL  = "claude-haiku-4-5-20251001"
client = anthropic.Anthropic()


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %B %d %Y at %I:%M %p")


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "get_current_time",
        "description": "Returns the current date and time. Use when the user asks what time or date it is.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "calculate",
        "description": "Evaluates a math expression. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression e.g. '15 * 4'"}
            },
            "required": ["expression"]
        }
    },
]


def run_tool(name: str, input_args: dict) -> str:
    if name == "get_current_time":
        return get_current_time()
    elif name == "calculate":
        return calculate(input_args["expression"])
    return f"Unknown tool: {name}"


DOCS_FOLDER = pathlib.Path(__file__).parent.parent / "docs"


def load_documents_from_folder(folder: pathlib.Path) -> list[str]:
    chunks = []
    for file in sorted(folder.glob("*")):
        if file.suffix == ".txt":
            text = file.read_text(encoding="utf-8")
            chunks.extend(p.strip() for p in text.split("\n\n") if p.strip())
        elif file.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file))
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    chunks.extend(p.strip() for p in page_text.split("\n\n") if p.strip())
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")
    return chunks


@st.cache_resource
def build_knowledge_base():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    db = chromadb.EphemeralClient()
    collection = db.get_or_create_collection("ai_knowledge", embedding_function=embedding_fn)
    documents = load_documents_from_folder(DOCS_FOLDER)
    if not documents:
        st.warning(f"No documents found in {DOCS_FOLDER}. Add .txt or .pdf files there.")
        return collection
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )
    return collection


def retrieve_context(collection, question: str, n: int = 3) -> str:
    results = collection.query(query_texts=[question], n_results=n)
    return "\n\n".join(f"- {d}" for d in results["documents"][0])


def chat(user_message: str, history: list, use_rag: bool, collection) -> str:
    if use_rag:
        context = retrieve_context(collection, user_message)
        content = (
            f"Use the context below to help answer. "
            f"If the answer isn't there, use your own knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_message}"
        )
    else:
        content = user_message

    history.append({"role": "user", "content": content})

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        tools=TOOLS,
        messages=history[-20:],
    )

    while response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        tool_result = run_tool(tool_block.name, tool_block.input)
        history.append({"role": "assistant", "content": response.content})
        history.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": tool_result}]
        })
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            tools=TOOLS,
            messages=history[-20:],
        )

    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply


st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("AI Chatbot")
st.caption("Built with Claude · Conversation Memory · Tool Use · RAG")

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle(
        "RAG mode",
        value=False,
        help="When ON: Claude answers using the docs/ folder. When OFF: Claude uses its own knowledge."
    )
    if use_rag:
        doc_count = build_knowledge_base().count()
        st.info(f"RAG is ON — {doc_count} chunks loaded from docs/ folder.")
    else:
        st.info("RAG is OFF — Claude answers from its own knowledge.")

    st.divider()
    st.markdown("**What this chatbot can do:**")
    st.markdown("- Remember conversation history")
    st.markdown("- Check current time")
    st.markdown("- Do math")
    st.markdown("- Answer from your docs (RAG mode)")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

collection = build_knowledge_base()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = chat(prompt, st.session_state.history, use_rag, collection)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
