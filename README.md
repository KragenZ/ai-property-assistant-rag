# 🏠 AI Property Assistant (LLM + RAG)

An intelligent chatbot built with Streamlit, LangChain, and Google Gemini. This assistant allows users to search a real-estate dataset using natural language. It leverages **Retrieval-Augmented Generation (RAG)** to fetch the most relevant properties from a local FAISS vector database and uses an LLM to explain why the properties match the user's criteria.

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
