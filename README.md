# RAG проект на LangChain

Простой RAG (Retrieval-Augmented Generation) на LangChain: загрузка документов, чанкинг, эмбеддинги, векторное хранилище Chroma и ответы на вопросы по контексту.

## Установка

```bash
cd c:\Users\anna\Desktop\RAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

При первом запуске скачается модель эмбеддингов (paraphrase-multilingual-MiniLM-L12-v2).

## Структура

- `src/documents.py` — загрузка PDF/TXT/DOCX и разбиение на чанки
- `src/rag.py` — эмбеддинги, Chroma и RAG-цепочка
- `main.py` — индексация документов и консольные вопросы
- `app.py` — **веб-интерфейс** (чат) для вопросов по документам
- `data/` — положите сюда свои документы для индексации

## Использование

1. Положите документы (PDF, TXT, DOCX) в папку `data/`.
2. Запуск:

**Вариант А — веб-интерфейс (чат):**

```bash
python -m streamlit run app.py
```

(Если команда `streamlit` не находится, используйте именно `python -m streamlit run app.py` — так запуск идёт через текущий интерпретатор Python.)

Откроется браузер с формой для вопросов. В боковой панели можно обновить индекс после добавления новых файлов в `data/`.

**Вариант Б — консоль:**

```bash
python main.py
```

Скрипт при первом запуске проиндексирует документы из `data/` (если база пуста). Для генерации ответов по умолчанию используется заглушка; чтобы подключить реальный LLM (OpenAI, Ollama и т.д.), измените `get_llm()` в `main.py` и при необходимости добавьте пакет в `requirements.txt`.

## Опционально: OpenAI

Раскомментируйте в `requirements.txt` строки с `langchain-openai` и `python-dotenv`, создайте `.env` с `OPENAI_API_KEY=...` и в `main.py` замените заглушку на:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### Эмбеддинги без PyTorch (обход WinError 1114)

Если на Windows возникает ошибка **WinError 1114** при загрузке PyTorch (c10.dll), можно не использовать локальные модели:

1. Установите `langchain-openai` и `python-dotenv`, создайте `.env` с `OPENAI_API_KEY=...`.
2. В `.env` добавьте строку: `USE_OPENAI_EMBEDDINGS=1`.
3. Эмбеддинги будут считаться через API OpenAI — PyTorch не потребуется.

## Ошибка WinError 1114 (PyTorch DLL) на Windows

Если при запуске появляется `OSError: [WinError 1114]` при загрузке `c10.dll` или других DLL PyTorch:

1. **Вариант А — переустановить PyTorch (CPU):**
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Вариант Б — установить Visual C++ Redistributable:**  
   Скачайте и установите [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), затем перезапустите терминал и снова запустите проект.

3. **Вариант В — не использовать PyTorch:**  
   Включите эмбеддинги через OpenAI (см. выше «Эмбеддинги без PyTorch»). Тогда локальный PyTorch не нужен.
