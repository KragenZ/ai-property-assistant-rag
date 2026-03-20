# 🏠 AI Property Assistant (LLM + RAG)

An intelligent chatbot built with Streamlit, LangChain, and Google Gemini. This assistant allows users to search a real-estate dataset using natural language. It leverages **Retrieval-Augmented Generation (RAG)** to fetch the most relevant properties from a local FAISS vector database and uses an LLM to explain why the properties match the user's criteria.

## 🎥 Application Demo
![App Demo](assets/demo.webp)

## 🚀 Features
- **Natural Language Search**: Ask questions like *"Find me a 2BHK under $200k in the CollgCr neighborhood."*
- **RAG Architecture**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) to embed property descriptions into a FAISS vector store. 
- **LLM Reasoning**: Uses Google's `gemini-2.5-flash` model to analyze the retrieved properties and formulate a conversational response.
- **Interactive UI**: Built entirely in Python using Streamlit for an easy-to-use chat interface.

## 🛠 Tech Stack
- **Python** 3.x
- **Streamlit** (Frontend Chat UI)
- **LangChain** (RAG Orchestration)
- **FAISS** (Local Vector Database)
- **HuggingFace Sentence Transformers** (Embeddings)
- **Google Generative AI / Gemini** (LLM)

## 🧠 Architecture Flow
1. **Data Ingestion**: A pre-processed real-estate dataset (`housing_clean.csv`) is loaded. Each row is converted into a rich text description.
2. **Embedding & Indexing**: The `all-MiniLM-L6-v2` model converts these descriptions into vector embeddings, which are stored locally using **FAISS**.
3. **Retrieval**: When a user asks a question, their query is embedded. FAISS performs a similarity search to retrieve the top 5 most relevant properties.
4. **Generation**: The user's query and the retrieved properties are passed as context to **Google Gemini** via a custom Prompt Template. Gemini generates a conversational, reasoning-based response.

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd ai-property-assistant-rag
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**3. Configure API Keys**
Create a `.env` file in the root of the project and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**4. Generate the FAISS Vector Database**
Before running the app, you need to generate embeddings for the dataset.
```bash
python src/vectorstore.py
```
*This will read `data/housing_clean.csv`, generate descriptions, and output the FAISS index to `faiss_index/`.*

**5. Start the Streamlit Application**
```bash
streamlit run app.py
```
