import streamlit as st
import os
import sys
import time

# Add src to path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from rag_engine import get_rag_chain

st.set_page_config(page_title="AI Property Assistant", page_icon="🏠", layout="wide")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .pipeline-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem; border-radius: 12px; color: white;
        font-family: monospace; font-size: 0.95rem; margin-bottom: 1rem;
        text-align: center; letter-spacing: 1px;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 0.75rem 1rem;
        border-left: 4px solid #667eea; margin-bottom: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .retrieved-card {
        background: #eef2ff; border-radius: 8px; padding: 0.75rem 1rem;
        margin-bottom: 0.5rem; border: 1px solid #c7d2fe;
    }
    .example-btn { cursor: pointer; }
    .stChatMessage { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("🏠 AI Property Assistant")
st.markdown("*Natural language property search powered by **RAG + Gemini***")

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ RAG Pipeline")
    st.markdown("""
    <div class="pipeline-box">
        📝 User Query<br>↓<br>
        🔢 Embed (MiniLM)<br>↓<br>
        🔍 FAISS Search<br>↓<br>
        📄 Top-K Context<br>↓<br>
        🤖 Gemini LLM<br>↓<br>
        💬 Answer + Reason
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("💡 Example Queries")
    examples = [
        "Find me a cheap house under $150k",
        "Best houses with 3 bedrooms in NAmes",
        "Find a 2BHK under $200k in CollgCr",
        "Largest house I can get in Sawyer?",
        "Show me houses newer than 10 years old",
    ]
    for ex in examples:
        if st.button(f"▶ {ex}", key=ex, use_container_width=True):
            st.session_state["prefill_query"] = ex

    st.divider()
    st.header("📊 Session Stats")
    num_queries = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.metric("Queries Asked", num_queries)
    avg_score = st.session_state.get("avg_relevance", None)
    if avg_score:
        st.metric("Avg Relevance Score", f"{avg_score:.0%}")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I am your AI Property Assistant. What kind of property are you looking for today?"
        }]
        st.session_state.pop("avg_relevance", None)
        st.rerun()

# ─── Load Engine ────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    try:
        return get_rag_chain()
    except Exception as e:
        return None, None

engine_result = load_engine()
chain, retriever = (engine_result if engine_result else (None, None))

# ─── Main Chat UI ────────────────────────────────────────────────────────────
if chain is None:
    st.warning("⚠️ Vectorstore not found or API key missing. Run `python src/vectorstore.py` first and configure `.env`.")
else:
    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I am your AI Property Assistant. What kind of property are you looking for today?"
        }]
    if "relevance_scores" not in st.session_state:
        st.session_state.relevance_scores = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show retrieval panel if stored
            if message["role"] == "assistant" and "retrieved" in message:
                with st.expander(f"🔍 Top {len(message['retrieved'])} Retrieved Properties (FAISS Vector Search)", expanded=False):
                    for i, doc in enumerate(message["retrieved"]):
                        st.markdown(f'<div class="retrieved-card"><b>Match #{i+1}</b>: {doc.page_content}</div>', unsafe_allow_html=True)
                if "relevance" in message:
                    score = message["relevance"]
                    color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                    st.caption(f"{color} Relevance Score: **{score:.0%}** (keyword overlap between query & retrieved context)")

    # React to user input (either from chat box or sidebar button)
    prefill = st.session_state.pop("prefill_query", None)
    prompt = st.chat_input("e.g. Find me a 3-bedroom house under $150k...") or prefill

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching vector database and generating response..."):
                try:
                    t0 = time.time()
                    retrieved_docs = retriever.invoke(prompt)
                    retrieval_time = time.time() - t0

                    # Build conversation history context
                    history = ""
                    for m in st.session_state.messages[-6:]:  # last 3 turns
                        role = "User" if m["role"] == "user" else "Assistant"
                        history += f"{role}: {m['content']}\n"

                    # Inject history into query for memory
                    augmented_prompt = f"[Conversation History]\n{history}\n[Current Query]\n{prompt}"
                    response = chain.invoke(augmented_prompt)

                    gen_time = time.time() - t0 - retrieval_time

                    st.markdown(response)

                    # ── Simple relevance evaluation ────────────────────────
                    query_words = set(prompt.lower().split())
                    context_text = " ".join(d.page_content.lower() for d in retrieved_docs)
                    context_words = set(context_text.split())
                    overlap = len(query_words & context_words)
                    score = min(overlap / max(len(query_words), 1), 1.0)
                    # Normalise: good if >30% keywords match
                    normalized_score = min(score * 3, 1.0)

                    st.session_state.relevance_scores.append(normalized_score)
                    st.session_state.avg_relevance = sum(st.session_state.relevance_scores) / len(st.session_state.relevance_scores)

                    # ── Show retrieval panel ───────────────────────────────
                    with st.expander(f"🔍 Top {len(retrieved_docs)} Retrieved Properties (FAISS Vector Search)", expanded=True):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f'<div class="retrieved-card"><b>Match #{i+1}</b>: {doc.page_content}</div>', unsafe_allow_html=True)

                    # ── Evaluation metrics row ─────────────────────────────
                    col1, col2, col3 = st.columns(3)
                    color = "🟢" if normalized_score >= 0.7 else "🟡" if normalized_score >= 0.4 else "🔴"
                    col1.metric("Relevance Score", f"{normalized_score:.0%}", help="Keyword overlap between query and retrieved context")
                    col2.metric("Retrieval Time", f"{retrieval_time:.2f}s")
                    col3.metric("Generation Time", f"{gen_time:.2f}s")
                    st.caption(f"{color} Relevance Score: **{normalized_score:.0%}** — measures how well the retrieved properties match your query keywords.")

                    # Store in history for replay
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "retrieved": retrieved_docs,
                        "relevance": normalized_score,
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}. Please check your Gemini API key in `.env`.")
