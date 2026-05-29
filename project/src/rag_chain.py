from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# В версии 0.3.x это самый надежный способ импорта:
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.vector_store import VectorDB

def get_rag_chain(config, vector_db: VectorDB):
    # 1. Загрузка базы
    vector_store = vector_db.load_local_db()
    if not vector_store:
        return None

    # 2. Настройка ретривера (поисковика)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config['search_k']}
    )

    # 3. Инициализация LLM (Qwen 2.5)
    llm = ChatOllama(
        model=config['llm_model'],
        base_url=config['ollama_base_url'],
        temperature=0.1,
        # num_ctx передается в ollama, но лучше зафиксировать это в Modelfile
        keep_alive="5m",
        num_ctx = 4096,  # Размер контекста (влияет на память)
        num_gpu = 999,  # Число слоев для переноса на GPU (999 = все слои)
    )

    # 4. Системный промпт
    system_prompt = (
        "Ты — умный помощник для работы с образовательными документами. "
        "Используй приведенный ниже контекст, чтобы ответить на вопрос пользователя. "
        "Если ответ не содержится в контексте, скажи: 'В документах нет информации об этом'. "
        "Не выдумывай информацию. "
        "Отвечай ВСЕГДА на русском языке, даже если контекст на английском.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Сборка цепочки (LCEL)
    # chain для обработки документов
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # chain для полного цикла (поиск + ответ)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain