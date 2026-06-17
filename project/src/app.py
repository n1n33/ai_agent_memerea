import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.config import load_config


config = load_config()
API_BASE_URL = os.getenv("RAG_API_URL", config.get("api_base_url", "http://localhost:8000")).rstrip("/")


class ApiError(RuntimeError):
    pass


def api_request(path: str, method: str = "GET", payload: dict | None = None, timeout: float = 30.0):
    url = f"{API_BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        raise ApiError(f"API вернул {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ApiError(f"API недоступен: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ApiError("API не ответил вовремя") from exc


def render_api_help():
    st.error(f"Не удалось подключиться к API по адресу `{API_BASE_URL}`.")
    with st.expander("Как запустить API"):
        st.code("uvicorn src.api:app --reload", language="powershell")


st.set_page_config(
    page_title=config["app_name"],
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stSpinner {text-align: center;}
</style>
""",
    unsafe_allow_html=True,
)

st.title(f"🎓 {config['app_name']}")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Управление базой знаний")
    st.caption(f"API: {API_BASE_URL}")

    try:
        health = api_request("/health", timeout=5)
        st.info(
            "\n".join(
                [
                    f"LLM: {health['llm_model']}",
                    f"Ollama: {health['ollama']}",
                    f"Vector DB: {health['vector_db']}",
                ]
            )
        )
    except ApiError:
        health = None
        render_api_help()

    if st.button("Пересобрать базу знаний", type="primary", disabled=health is None):
        with st.status("Обновление индекса...", expanded=True) as status:
            try:
                st.write("API читает документы и пересобирает FAISS-индекс...")
                result = api_request("/rebuild", method="POST", timeout=600)
                status.update(label="Готово", state="complete", expanded=False)
                st.success(f"База обновлена. Документов загружено: {result['documents_loaded']}")
            except ApiError as exc:
                status.update(label="Ошибка", state="error")
                st.error(str(exc))

    st.divider()
    st.markdown("### Загруженные файлы")
    try:
        files_response = api_request("/files", timeout=10)
        files = files_response.get("files", [])
        if files:
            for file_name in files:
                st.caption(f"📄 {file_name}")
        else:
            st.caption("Файлы не найдены")
    except ApiError as exc:
        st.caption(str(exc))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Задайте вопрос по лекциям или документам...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Qwen изучает материалы..."):
            try:
                response = api_request(
                    "/predict",
                    method="POST",
                    payload={"question": prompt},
                    timeout=300,
                )
            except ApiError as exc:
                error_message = str(exc)
                st.error(error_message)
                if "Ollama" in error_message:
                    with st.expander("Команды для проверки Ollama"):
                        st.code(
                            f"""ollama serve
ollama list
ollama run {config['llm_model']} "привет\"""",
                            language="powershell",
                        )
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.stop()

        answer = response["answer"]
        st.markdown(answer)

        sources = response.get("sources", [])
        if sources:
            with st.expander("Использованные источники"):
                for source in sources:
                    st.markdown(f"- **{source}**")

        elapsed_seconds = response.get("elapsed_seconds")
        if elapsed_seconds is not None:
            st.caption(f"Время генерации: {elapsed_seconds:.2f} сек.")

        st.session_state.messages.append({"role": "assistant", "content": answer})
