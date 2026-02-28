"""
RAG: эмбеддинги, векторное хранилище и цепочка вопрос–ответ.
"""
from pathlib import Path
from typing import List, Any

from langchain_core.documents import Document

# Локальная модель для эмбеддингов (используется, если не включены OpenAI-эмбеддинги)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _OpenAIEmbeddingsNoTorch:
    """Эмбеддинги через API OpenAI без импорта langchain_openai (и без PyTorch)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        out = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend([item.embedding for item in sorted(resp.data, key=lambda x: x.index)])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Создать модель эмбеддингов.
    Если заданы USE_OPENAI_EMBEDDINGS=1 и OPENAI_API_KEY — используются OpenAI (без PyTorch).
    Иначе — локальные HuggingFace (требуют PyTorch).
    """
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("USE_OPENAI_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            return _OpenAIEmbeddingsNoTorch(api_key=api_key, model="text-embedding-3-small")
    # Ленивый импорт, чтобы не грузить PyTorch при использовании OpenAI-эмбеддингов
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    except OSError as e:
        if "1114" in str(e) or "DLL" in str(e).upper() or "c10" in str(e).lower():
            raise RuntimeError(
                "Не удалось загрузить PyTorch (ошибка DLL на Windows). "
                "Варианты: 1) pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cpu "
                "2) Установить Visual C++ Redistributable. "
                "3) В .env задать USE_OPENAI_EMBEDDINGS=1 и OPENAI_API_KEY=..."
            ) from e
        raise


def get_vector_store(
    persist_directory: str | Path,
    embedding_function=None,
    collection_name: str = "rag_docs",
):
    """Создать или подключиться к векторному хранилищу Chroma."""
    from langchain_chroma import Chroma
    if embedding_function is None:
        embedding_function = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
    )


def build_rag_chain(vector_store: Any, llm, k: int = 4):
    """
    Собрать RAG без импорта PromptTemplate/StrOutputParser (они тянут PyTorch → краш 0xC0000005).
    llm — объект с методом .invoke(x), возвращающий строку.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt_tpl = (
        "Ты помощник, отвечающий на вопросы по предоставленному контексту. "
        "Отвечай только на основе контекста ниже. Если ответа в контексте нет — так и скажи.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    )

    class _RAGChain:
        def invoke(self, question: str) -> str:
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            text = prompt_tpl.format(context=context, question=question)
            out = llm.invoke(text)
            return out if isinstance(out, str) else getattr(out, "content", str(out))

    return _RAGChain()


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _query_words(text: str) -> set:
    """Ключевые слова запроса для гибридного поиска (слова от 3 букв)."""
    import re
    words = re.findall(r"[а-яёa-z]{3,}", text.lower())
    return set(words)


def _keyword_score(chunk_text: str, words: set) -> float:
    """Доля слов запроса, встретившихся в чанке (0..1)."""
    if not words:
        return 0.0
    chunk_lower = chunk_text.lower()
    found = sum(1 for w in words if w in chunk_lower)
    return found / len(words)


def build_inmemory_rag_chain(documents_dir: Path, llm, k: int = 10, chunk_size: int = 1000, chunk_overlap: int = 200, hybrid: bool = True):
    """
    RAG без Chroma: документы из папки, эмбеддинги в памяти.
    hybrid=True: комбинация семантического поиска и совпадения ключевых слов (лучше для вопросов по законам).
    """
    from src.documents import load_documents_from_folder, split_documents

    docs = load_documents_from_folder(documents_dir)
    if not docs:
        raise FileNotFoundError(f"В {documents_dir} нет .pdf, .txt или .docx")
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings_fn = get_embeddings()
    print(f"  эмбеддинг {len(chunks)} чанков...", flush=True)
    texts = [c.page_content for c in chunks]
    vectors = embeddings_fn.embed_documents(texts)
    stored = list(zip(vectors, chunks))
    prompt_tpl = (
        "Ты юридический помощник. Ответь на вопрос пользователя строго на основе приведённого ниже контекста из документов. "
        "Учитывай, что в тексте могут быть синонимы (например, «увольнение» — то же, что «уволить»; «заработная плата» — то же, что «зарплата»). "
        "Если в контексте есть ответ — сформулируй его ясно и укажи, что это по приведённым документам. Если однозначного ответа нет — так и скажи.\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    )

    class _InMemoryRAGChain:
        def invoke(self, question: str) -> str:
            q_emb = embeddings_fn.embed_query(question)
            words = _query_words(question) if hybrid else set()
            # Семантика + доля ключевых слов в чанке
            scored = []
            for v, d in stored:
                sim = _cosine_sim(q_emb, v)
                kw = _keyword_score(d.page_content, words) if words else 0.0
                combined = sim + 0.25 * kw
                scored.append((combined, d))
            scored.sort(key=lambda x: -x[0])
            top = [d for _, d in scored[:k]]
            context = "\n\n".join(d.page_content for d in top)
            text = prompt_tpl.format(context=context, question=question)
            out = llm.invoke(text)
            return out if isinstance(out, str) else getattr(out, "content", str(out))

    return _InMemoryRAGChain()
