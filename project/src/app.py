import streamlit as st
import time
from urllib.request import urlopen
from src.config import load_config
from src.document_loader import DocumentLoader
from src.vector_store import VectorDB
from src.rag_chain import get_rag_chain
from pathlib import Path


def check_ollama_connection(base_url: str, timeout: float = 2.0):
    api_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(api_url, timeout=timeout) as response:
            return 200 <= response.status < 300, ""
    except Exception as exc:
        return False, str(exc)


# Загрузка настроек
config = load_config()


SUPPORTED_DATA_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".raw"}


def dataset_files_exist(data_path):
    path = Path(data_path)
    if not path.exists():
        return False

    return any(
        file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
        for file_path in path.rglob("*")
    )


def ensure_dataset_exists(data_path):
    if dataset_files_exist(data_path):
        return

    with st.spinner("Данные не найдены. Запускаю автоматическую загрузку датасета..."):
        try:
            from data.download_data import main as download_data

            download_data()
        except Exception as exc:
            st.error(f"Не удалось автоматически загрузить данные: {exc}")
            st.stop()

    if not dataset_files_exist(data_path):
        st.error(f"После загрузки данные не найдены в папке: {data_path}")
        st.stop()

    st.success("Данные загружены автоматически.")


st.set_page_config(
    page_title=config['app_name'],
    page_icon="🎓",
    layout="wide"
)
ensure_dataset_exists(config['data_path'])

# Стилизация
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stSpinner {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title(f"🎓 {config['app_name']}")

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- САЙДБАР (Настройки и База) ---
with st.sidebar:
    st.header("⚙️ Управление базой знаний")

    st.info(f"LLM: {config['llm_model']}\nDevice: {config['embedding_device'].upper()}")

    if st.button("🔄 Пересобрать базу знаний", type="primary"):
        with st.status("Обновление индекса...", expanded=True) as status:
            st.write("📂 Чтение файлов...")
            loader = DocumentLoader(config['data_path'])
            docs = loader.load_documents()

            if docs:
                st.write(f"🧩 Разбиение на чанки и векторизация ({len(docs)} док.)...")
                vdb = VectorDB(config)
                vdb.create_vector_db(docs)
                status.update(label="Готово!", state="complete", expanded=False)
                st.success(f"База обновлена! Всего документов: {len(docs)}")
            else:
                status.update(label="Ошибка", state="error")
                st.error("Файлы не найдены в папке data/raw")

    st.divider()
    st.markdown("### Загруженные файлы:")
    # Простое отображение списка файлов, если база существует
    try:
        import os

        files = os.listdir(config['data_path'])
        if files:
            for f in files:
                st.caption(f"📄 {f}")
        else:
            st.caption("Папка пуста")
    except:
        pass

# --- ЧАТ ---
# Отрисовка истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод пользователя
if prompt := st.chat_input("Задайте вопрос по лекциям или документам..."):
    # Добавляем в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        # Генерация ответа
        with st.chat_message("assistant"):
            ollama_ok, ollama_error = check_ollama_connection(config['ollama_base_url'])
            if not ollama_ok:
                error_message = (
                    f"Не удалось подключиться к Ollama по адресу `{config['ollama_base_url']}`. "
                    "Запустите Ollama и повторите запрос."
                )
                st.error(error_message)
                with st.expander("Команды для проверки Ollama"):
                    st.code(
                        f"""ollama serve
    ollama list
    ollama run {config['llm_model']} "привет\"""",
                        language="powershell",
                    )
                    if ollama_error:
                        st.caption(f"Техническая ошибка: {ollama_error}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.stop()

            vdb = VectorDB(config)
            rag_chain = get_rag_chain(config, vdb)

        if rag_chain:
            start_time = time.time()
            with st.spinner("Qwen изучает материалы..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    context = response['context']

                    # Вывод ответа
                    st.markdown(answer)

                    # Блок с источниками (Expander)
                    with st.expander("📚 Использованные источники"):
                        seen_sources = set()
                        for doc in context:
                            source = doc.metadata.get('source_file', 'Неизвестный файл')
                            page = doc.metadata.get('page', 'Неизвестная стр.')  # Для PDF

                            # Формируем уникальную строку источника
                            source_info = f"{source}"
                            if 'page' in doc.metadata:
                                source_info += f" (стр. {page + 1})"

                            if source_info not in seen_sources:
                                st.markdown(f"- **{source_info}**")
                                # Можно показать фрагмент текста, если нужно:
                                # st.caption(doc.page_content[:200] + "...")
                                seen_sources.add(source_info)

                    elapsed = time.time() - start_time
                    st.caption(f"⏱️ Время генерации: {elapsed:.2f} сек.")

                    # Сохраняем ответ ассистента в историю
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Произошла ошибка при генерации: {e}")
        else:
            st.warning("⚠️ База знаний не найдена. Пожалуйста, нажмите 'Пересобрать базу знаний' в меню слева.")
