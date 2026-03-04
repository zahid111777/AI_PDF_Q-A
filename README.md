# AI PDF Q&A

An intelligent PDF question-answering system powered by LangChain, OpenAI, and hybrid retrieval (BM25 + semantic search). Upload a PDF and ask natural language questions about its content.

## ✨ Features

- **🔍 Hybrid Retrieval**: Combines BM25 keyword search with semantic similarity via Qdrant vector store
- **💬 Conversational QA**: Maintains chat history and context for natural conversations
- **🧠 Semantic Understanding**: Uses OpenAI embeddings for deep semantic search
- **📄 PDF Processing**: Automatic PDF loading with configurable chunk size and overlap
- **🎨 Streamlit UI**: Clean, responsive web interface for easy interaction
- **⚙️ Configurable**: Adjustable model selection, retrieval weights, and parameters

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o / GPT-4o-mini / GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: Qdrant (cloud-based)
- **Retrieval**: LangChain EnsembleRetriever (BM25 + semantic)
- **Frontend**: Streamlit
- **PDF Processing**: PyPDF, LangChain RecursiveCharacterTextSplitter
- **Backend**: Python 3.x with LangChain

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Qdrant Cloud account (for vector store)

## 🚀 Installation

1. **Clone the repository** (or navigate to the project folder)

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** in `keys/.env`:
   ```
   QDRANT_URL=https://your-qdrant-cloud-url
   api_key=your-qdrant-api-key
   OPENAI_API_KEY=your-openai-api-key
   ```

## 🎯 Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run frontend/frontend.py
   ```

2. **Upload a PDF**: Use the sidebar to upload your PDF file

3. **Ask Questions**: Type your questions in the chat interface

4. **View Sources**: Each answer includes retrieved source chunks with page numbers

5. **Adjust Retrieval**: Use sidebar controls to tune BM25/Qdrant weights and select LLM model

## 📁 Project Structure

```
AI_PDF_QA/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── backend/
│   └── src/
│       └── main.py           # Core QA pipeline & retrieval logic
├── frontend/
│   └── frontend.py           # Streamlit web UI
└── keys/
    └── .env                  # Environment variables (excluded from git)
```

### Core Components

- **PDFProcessor**: Loads PDFs and splits into overlapping chunks
- **VectorStoreManager**: Manages Qdrant vector store with OpenAI embeddings
- **BM25Retriever**: Keyword-based retrieval using rank-bm25
- **EnsembleRetriever**: Combines BM25 + semantic search with configurable weights
- **QAPipeline**: Builds the complete RAG chain with conversation history

## ⚙️ Configuration

### LLM Model Selection
- `gpt-4o-mini`: Fastest, most cost-effective
- `gpt-4o`: Balance of speed and quality
- `gpt-3.5-turbo`: Legacy, faster but lower quality

### Retrieval Tuning
- Adjust BM25 and Qdrant weights to balance keyword vs. semantic search
- Default chunk size: 1000 characters with 200 character overlap
- Retrieval returns top 4 most relevant chunks by default

## 🔐 Security Notes

- Never commit the `keys/.env` file to version control
- Keep API keys secure and rotate periodically
- Use environment variables for all sensitive configuration

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]