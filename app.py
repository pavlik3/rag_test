"""
Веб-интерфейс для RAG: чат с вопросами по документам.
Режим «в памяти» — без Chroma, без PyTorch. Запуск: python -m streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from main import DATA_DIR, get_llm
from src.rag import get_embeddings, build_inmemory_rag_chain
from src.documents import load_documents_from_folder, split_documents


@st.cache_resource
def get_rag_chain():
    """Один раз загружаем документы из data/, эмбеддим в памяти, собираем цепочку. Без Chroma."""
    try:
        chain = build_inmemory_rag_chain(DATA_DIR, get_llm(), k=10)
        docs = load_documents_from_folder(DATA_DIR)
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        return chain, len(chunks)
    except FileNotFoundError:
        return None, 0
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить RAG: {e}") from e


def main():
    st.set_page_config(page_title="RAG — вопросы по документам", page_icon="📄")
    st.title("📄 RAG: вопросы по вашим документам")
    st.caption("Режим в памяти (документы из папки data/). Без Chroma.")

    with st.sidebar:
        st.header("Документы")
        if st.button("Перезагрузить документы из data/"):
            st.cache_resource.clear()
            st.rerun()
        st.caption("Положите PDF, TXT или DOCX в папку **data/** и нажмите кнопку.")

    try:
        chain, num_chunks = get_rag_chain()
    except Exception as e:
        import traceback
        st.error(f"Ошибка при загрузке RAG: {e}")
        st.code(traceback.format_exc())
        st.info("Проверьте .env: OPENAI_API_KEY и USE_OPENAI_EMBEDDINGS=1.")
        return

    if chain is None:
        st.warning("В папке **data/** нет документов (.pdf, .txt, .docx). Добавьте файлы и нажмите «Перезагрузить документы» в боковой панели.")
        return

    st.caption(f"Загружено чанков в памяти: {num_chunks}. Задайте вопрос ниже.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Вопрос по документам..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ищу в документах и формирую ответ..."):
                try:
                    answer = chain.invoke(prompt)
                except Exception as e:
                    answer = f"Ошибка: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages and st.sidebar.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        st.error(f"Ошибка при запуске: {e}")
        st.code(traceback.format_exc())
