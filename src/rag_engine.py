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
    # If vectorstore doesn't exist, we can't answer (should be handled in app)
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vectorstore not built yet.")
        return None
        
    print("Loading vectorstore...")
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Create the retriever (fetch top 5 matching properties)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Initialize the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    
    # Define our Prompt
    template = """You are an AI Property Assistant. Your job is to help the user find a suitable property based on their query.
You have retrieved the following property listings from our database that match the user's request. 

Context Properties Data:
{context}

User Query: {question}

Instructions:
1. Answer the user's query in a friendly and conversational way.
2. Recommend the best matching properties from the context provided above.
3. If the context doesn't contain exactly what they want, suggest the closest matches and explain why.
4. Provide a clear reason for why a property matches (e.g., "This 2BHK in CollgCr is a great match because it is priced at $150k...").
5. Do NOT make up properties that are not in the context.

Response formatting: use bullet points for listing properties. Be helpful and professional.
"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        output = []
        for i, doc in enumerate(docs):
            output.append(f"Property {i+1}:\n{doc.page_content}\n")
        return "\n".join(output)

    # Build the LangChain Runnable sequence
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever
