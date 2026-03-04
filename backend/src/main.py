"""
backend/src/main.py
AI PDF Q&A — Hybrid Retrieval Pipeline
Approach: BM25 (keyword) + Qdrant (semantic) via LangChain EnsembleRetriever
"""

import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ── Load .env ──────────────────────────────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "keys", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("api_key")
COLLECTION_NAME = "pdf_qa_collection"


# ── PDF Processor ─────────────────────────────────────────────────────────

class PDFProcessor:
    """Load a PDF and split it into overlapping text chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_and_split(self, pdf_path: str) -> List[Document]:
        """Return a list of Document chunks from the given PDF file path."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = self.splitter.split_documents(pages)
        print(f"[PDFProcessor] {len(pages)} pages → {len(chunks)} chunks")
        return chunks


# ── Vector Store Manager (Qdrant + OpenAI Embeddings) ────────────────────

class VectorStoreManager:
    """
    Build and manage a Qdrant cloud vector store with OpenAI embeddings.
    Each call to build() recreates the collection so the index is always
    fresh for the currently uploaded PDF.
    """

    def __init__(
        self,
        openai_api_key: str,
        qdrant_url: str = QDRANT_URL,
        qdrant_api_key: str = QDRANT_API_KEY,
        embed_model: str = "text-embedding-3-small",
        collection_name: str = COLLECTION_NAME,
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embed_model,
        )
        self.vector_store: Optional[QdrantVectorStore] = None

    def build(self, documents: List[Document]) -> "VectorStoreManager":
        """Embed and upsert all documents into Qdrant (recreates collection each time)."""
        self.vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            force_recreate=True,   # drop & recreate so old PDF data is wiped
            prefer_grpc=False,
        )
        print(
            f"[VectorStoreManager] Qdrant collection '{self.collection_name}' "
            f"indexed with {len(documents)} docs"
        )
        return self

    def get_retriever(self, k: int = 4):
        if self.vector_store is None:
            raise ValueError("Vector store has not been built yet. Call build() first.")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


# ── Hybrid Retriever (BM25 + Qdrant Ensemble) ────────────────────────────

class HybridRetriever:
    """
    Combines:
      - BM25Retriever    → keyword / sparse retrieval
      - QdrantRetriever  → dense / semantic retrieval
    via LangChain's EnsembleRetriever with configurable weights.
    """

    def __init__(
        self,
        documents: List[Document],
        vector_store_manager: VectorStoreManager,
        bm25_weight: float = 0.4,
        qdrant_weight: float = 0.6,
        k: int = 4,
    ):
        if abs(bm25_weight + qdrant_weight - 1.0) > 1e-6:
            raise ValueError("bm25_weight + qdrant_weight must equal 1.0")

        # BM25 — keyword retrieval
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = k

        # Qdrant — semantic retrieval
        qdrant = vector_store_manager.get_retriever(k=k)

        self.retriever = EnsembleRetriever(
            retrievers=[bm25, qdrant],
            weights=[bm25_weight, qdrant_weight],
        )
        print(
            f"[HybridRetriever] BM25 weight={bm25_weight} | "
            f"Qdrant weight={qdrant_weight} | k={k}"
        )

    def get(self):
        return self.retriever


# ── QA Engine (LCEL — LangChain 1.x compatible) ───────────────────────────

_QA_SYSTEM = """\
You are a helpful assistant that answers questions based strictly on the \
provided document context. If the answer is not in the context, say \
"I don't have enough information in the document to answer that."

Context:
{context}
"""

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in docs
    )


class QAEngine:
    """
    LCEL-based conversational QA engine.
    Maintains an in-memory chat history list for multi-turn conversations.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=temperature,
        )
        self.retriever = None
        self.chat_history: List = []

    def build_chain(self, retriever) -> None:
        """Attach the hybrid retriever."""
        self.retriever = retriever
        print("[QAEngine] LCEL chain ready.")

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Ask a question; returns (answer_text, source_documents)."""
        if self.retriever is None:
            raise ValueError("Chain not initialised. Call build_chain() first.")

        # Retrieve relevant chunks
        source_docs: List[Document] = self.retriever.invoke(question)
        context = _format_docs(source_docs)

        # Build and run the prompt
        chain = _QA_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context,
                "chat_history": self.chat_history,
                "question": question,
            }
        )

        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, source_docs

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
        print("[QAEngine] Memory cleared.")


# ── Public Factory Function ───────────────────────────────────────────────

def build_qa_pipeline(
    pdf_path: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    bm25_weight: float = 0.4,
    qdrant_weight: float = 0.6,
    k: int = 4,
    qdrant_url: str = QDRANT_URL,
    qdrant_api_key: str = QDRANT_API_KEY,
) -> QAEngine:
    """
    End-to-end factory:
      PDF → chunks → Qdrant (semantic) + BM25 (keyword) hybrid retriever → LCEL QA chain

    Returns a ready-to-use QAEngine.
    """
    # 1. Chunk the PDF
    processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = processor.load_and_split(pdf_path)

    if not documents:
        raise ValueError("No text could be extracted from the PDF.")

    # 2. Build Qdrant vector store
    vsm = VectorStoreManager(
        openai_api_key=api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )
    vsm.build(documents)

    # 3. Build hybrid retriever
    hybrid = HybridRetriever(
        documents=documents,
        vector_store_manager=vsm,
        bm25_weight=bm25_weight,
        qdrant_weight=qdrant_weight,
        k=k,
    )

    # 4. Build and return QA engine
    engine = QAEngine(openai_api_key=api_key, model=model)
    engine.build_chain(hybrid.get())

    return engine
