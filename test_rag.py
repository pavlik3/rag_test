"""
Один тестовый запрос к RAG — проверка поиска по документам и ответа.
Запуск: python test_rag.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import (
    DATA_DIR,
    PERSIST_DIR,
    COLLECTION_NAME,
    get_llm,
)
from src.rag import get_embeddings, get_vector_store, build_rag_chain, build_inmemory_rag_chain


def main():
    import sys
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    sys.stdout.reconfigure(encoding="utf-8")
    print("Загрузка RAG (режим в памяти, без Chroma)...", flush=True)
    llm = get_llm()
    try:
        chain = build_inmemory_rag_chain(DATA_DIR, llm, k=10)
    except FileNotFoundError as e:
        print(e, flush=True)
        return
    print("  цепочка готова\n", flush=True)

    question = "Что такое трудовой договор по Трудовому кодексу?"
    print("Вопрос:", question, flush=True)
    print("-" * 60, flush=True)
    print("Ищу в документах и вызываю LLM...", flush=True)
    try:
        answer = chain.invoke(question)
        print("Ответ:", answer, flush=True)
    except Exception as e:
        print("Ошибка:", e, flush=True)
    print("-" * 60, flush=True)
    print("Готово.", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("\nОшибка:", e, flush=True)
        traceback.print_exc()
