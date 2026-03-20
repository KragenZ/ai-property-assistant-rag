# 🏠 AI Property Assistant (LLM + RAG)

An intelligent property search chatbot built with **Streamlit**, **LangChain**, and **Google Gemini**. Users can search a real-estate dataset using natural language. The system uses **Retrieval-Augmented Generation (RAG)** to find semantically relevant properties and generate grounded, explainable answers.

## 🌐 Live Application
**[▶ Try it live on Streamlit](https://ai-property-assistant-rag-8za8yz32sjafid8qigpxdc.streamlit.app/)**

---

## ⚙️ How It Works

```
User Query
    │
    ▼
[Embed with all-MiniLM-L6-v2]   ← free local HuggingFace model
    │
    ▼
[FAISS Vector Search]            ← cosine similarity over 500 properties
    │
    ▼
[Top-5 Context Retrieved]        ← shown to user in the UI
    │
    ▼
[Gemini LLM + RAG Prompt]        ← property context injected into prompt
    │
    ▼
[Grounded Answer + Reasoning]    ← "This matches because..."
```

The system also evaluates each response with a **relevance score** (keyword overlap between the query and retrieved context) and shows retrieval + generation latency in real time.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 **Natural Language Search** | Ask anything: *"Find cheap houses with 3 bedrooms in NAmes"* |
| 🧠 **RAG Architecture** | FAISS + HuggingFace embeddings for fast vector retrieval |
| 🤖 **LLM Reasoning** | Gemini explains *why* each property matches your criteria |
| 💬 **Conversation Memory** | Follow-up questions aware of prior context |
| 📊 **Evaluation Metrics** | Relevance score, retrieval time, generation time per query |
| 🗂 **Retrieval Panel** | See exactly which properties were retrieved from the database |

---

## 💡 Example Queries
- `"Find me a cheap house under $150k"`
- `"Best houses with 3 bedrooms in NAmes"`
- `"Find a 2BHK under $200k in CollgCr"`
- `"Largest house I can get in Sawyer?"`
- `"Show me houses newer than 10 years old"`

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS (local) |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Dataset | Ames Housing Dataset (500 properties) |

---

## 📁 Project Structure

```
ai-property-assistant-rag/
├── app.py                  # Streamlit UI + RAG pipeline integration
├── src/
│   ├── vectorstore.py      # Data processing + FAISS index generation
│   └── rag_engine.py       # LLM chain, retriever, prompt template
├── data/
│   └── housing_clean.csv   # Pre-processed housing dataset
├── faiss_index/            # Generated FAISS vector store
├── requirements.txt
└── .env                    # API keys (not committed)
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/KragenZ/ai-property-assistant-rag.git
cd ai-property-assistant-rag
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**3. Configure API Keys**
```env
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

**4. Generate the FAISS Vector Database**
```bash
python src/vectorstore.py
```

**5. Run the Application**
```bash
streamlit run app.py
```
