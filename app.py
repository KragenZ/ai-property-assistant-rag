import streamlit as st
import os
import sys

# Add src to path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from rag_engine import get_rag_chain

st.set_page_config(page_title="AI Property Assistant", page_icon="🏠", layout="wide")

st.title("🏠 AI Property Assistant (LLM + RAG)")
st.markdown("Ask me to find properties, explore neighborhoods, or make recommendations based on your budget!")

# Load the RAG chain
@st.cache_resource
def load_engine():
    try:
        return get_rag_chain()
    except Exception as e:
        st.error(f"Error loading RAG Engine: {e}")
        return None, None

engine_result = load_engine()
if engine_result:
    chain, retriever = engine_result
else:
    chain, retriever = None, None

if chain is None:
    st.warning("⚠️ Welcome! It looks like the vectorstore hasn't been built yet, or your API key is missing. Ensure you run `python src/vectorstore.py` first and configure `.env`.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Property Assistant. What kind of property are you looking for today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Find me a 2BHK under $200k in CollgCr..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching the database and analyzing properties..."):
                try:
                    # Get retrieved docs to show in an expander
                    retrieved_docs = retriever.invoke(prompt)
                    
                    # Generate response
                    response = chain.invoke(prompt)
                    
                    st.markdown(response)
                    
                    with st.expander("🔍 See Retrieved Properties Data"):
                        for i, doc in enumerate(retrieved_docs):
                            st.info(f"**Match {i+1}**: {doc.page_content}")
                            st.json(doc.metadata)  # Show the raw metadata
                            
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please check your Gemini API key in the `.env` file.")
