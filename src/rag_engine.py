import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_rag_chain():
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vectorstore not built yet.")
        return None

    print("Loading vectorstore...")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

    # Create the retriever (fetch top 5 matching properties)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    # Improved prompt that handles conversation context and avoids markdown code blocks
    template = """You are a friendly and professional AI Property Assistant helping users find the right home.

Below are the top property listings retrieved from our database that are semantically similar to the user's query.

--- Retrieved Properties ---
{context}
----------------------------

{question}

Instructions:
1. Be conversational, friendly and concise.
2. Recommend the best 2-3 matching properties from the context above with a clear reason why each is a good match.
3. If the user is asking a follow-up question, refer to the conversation history provided.
4. If no properties match exactly, recommend the closest options and explain why.
5. Do NOT invent properties not listed above.
6. Use plain text bullet points only. Do NOT use code blocks or backtick formatting in your response.

Your response:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(
            f"Property {i+1}: {doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # Build the LangChain Runnable sequence
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
