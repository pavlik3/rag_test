"""
Точка входа: индексация документов из data/ и интерактивные вопросы по RAG.
"""
import sys
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.documents import load_documents_from_folder, split_documents
from src.rag import get_embeddings, get_vector_store, build_rag_chain, build_inmemory_rag_chain

# Папки по умолчанию
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "rag_docs"


def get_llm():
    """
    Возвращает LLM для генерации ответов.
    Если задан OPENAI_API_KEY в .env — используется OpenAI Chat (gpt-4o-mini) без PyTorch.
    Иначе — заглушка с подсказкой подключить LLM.
    """
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        class OpenAILLM:
            def __init__(self, key):
                self.key = key
            def invoke(self, x):
                import openai
                client = openai.OpenAI(api_key=self.key)
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": x}],
                    temperature=0,
                )
                return r.choices[0].message.content or ""
        return OpenAILLM(api_key)

    # Заглушка без импорта langchain_core.language_models (иначе подтягивается PyTorch)
    class EchoLLM:
        def invoke(self, x):
            return "Контекст передан в промпт. Задайте OPENAI_API_KEY в .env для ответов через GPT."

    return EchoLLM()


def index_documents():
    """Загрузить документы из data/, разбить на чанки и записать в Chroma."""
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir(parents=True)
        print(f"Создана папка {DATA_DIR}. Положите туда PDF/TXT/DOCX и запустите снова.")
        return False

    documents = load_documents_from_folder(DATA_DIR)
    if not documents:
        print(f"В {DATA_DIR} нет подходящих файлов (.pdf, .txt, .docx).")
        return False

    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    print(f"Загружено документов: {len(documents)} → разбито на чанков: {len(chunks)}")
    print(f"  (размер чанка: 1000 символов, перекрытие: 200)")

    embeddings = get_embeddings()
    vector_store = get_vector_store(PERSIST_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    vector_store.add_documents(chunks)
    print(f"Чанки сохранены в векторной БД: {PERSIST_DIR}")
    if chunks:
        sample = chunks[0].page_content[:200].replace("\n", " ")
        print(f"  Пример первого чанка: «{sample}...»")
    return True


def main():
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    embeddings = get_embeddings()
    llm = get_llm()
    # Режим «в памяти»: без Chroma, поиск по документам из data/. Не зависает.
    use_inmemory = os.environ.get("USE_INMEMORY_RAG", "1").strip().lower() in ("1", "true", "yes")
    if use_inmemory:
        print("Загрузка документов из data/ и построение поиска в памяти...", flush=True)
        try:
            chain = build_inmemory_rag_chain(DATA_DIR, llm, k=10)
            print("Готово.\n", flush=True)
        except FileNotFoundError as e:
            print(e)
            print("Положите PDF/TXT/DOCX в папку data/ и запустите снова.")
            return
    else:
        vector_store = get_vector_store(PERSIST_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
        chain = build_rag_chain(vector_store, llm, k=4)

    print("RAG готов. Задавайте вопросы (пустая строка — выход).\n")
    while True:
        try:
            q = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            answer = chain.invoke(q)
            print(f"Ответ: {answer}\n")
        except Exception as e:
            print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
