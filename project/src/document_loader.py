import os
from glob import glob
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
from loguru import logger


class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        documents = []
        # Поддерживаемые расширения
        extensions = {
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'txt': TextLoader,
            'md': TextLoader,  # TextLoader отлично читает MD
            'raw': TextLoader
        }

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            logger.warning(f"Created missing directory: {self.data_path}")
            return []

        files_found = 0

        for ext, loader_cls in extensions.items():
            pattern = os.path.join(self.data_path, f"*.{ext}")
            files = glob(pattern)

            for file_path in files:
                try:
                    if loader_cls == TextLoader:
                        loader = loader_cls(file_path, encoding='utf-8')
                    else:
                        loader = loader_cls(file_path)

                    docs = loader.load()

                    # Добавляем метаданные, чтобы знать источник
                    for doc in docs:
                        doc.metadata['source_file'] = os.path.basename(file_path)

                    documents.extend(docs)
                    files_found += 1
                    logger.info(f"Loaded: {os.path.basename(file_path)}")

                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.success(f"Total files loaded: {files_found}")
        return documents