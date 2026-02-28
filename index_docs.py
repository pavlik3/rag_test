"""
Только индексация документов из data/ в chroma_db. Без проверки count() и без чата.
Запуск: python index_docs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import index_documents

if __name__ == "__main__":
    print("Индексация документов из data/ ...", flush=True)
    if index_documents():
        print("Готово. Дальше: python main.py", flush=True)
    else:
        print("Положите PDF/TXT/DOCX в папку data/ и запустите снова.", flush=True)
