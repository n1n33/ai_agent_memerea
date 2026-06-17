import os
import torch
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger


class VectorDB:
    def __init__(self, config):
        self.config = config
        self.vector_store_path = config['vector_store_path']

        # Настройка устройства для эмбеддингов
        device = config.get('embedding_device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available! Switching to CPU.")
            device = 'cpu'

        logger.info(f"Initializing Embeddings on {device}...")

        # E5 модели требуют префикс "query: " для запросов, но LangChain HF Embeddings
        # обычно хэндлит это через encode_kwargs, либо мы используем модель "как есть".
        # Для E5 важно normalize_embeddings=True.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config['embedding_model'],
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vector_db(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents provided for indexing.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            separators=["\n\n", "\n", " ", ""]
        )

        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} chunks from documents.")

        if not texts:
            return None

        # Создаем FAISS индекс
        vector_store = FAISS.from_documents(texts, self.embeddings)

        # Сохраняем локально
        vector_store.save_local(self.vector_store_path)
        logger.success(f"Vector DB saved to {self.vector_store_path}")
        return vector_store

    def load_local_db(self):
        if not os.path.exists(self.vector_store_path):
            return None

        try:
            vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            logger.error(f"Error loading Vector DB: {e}")
            return None