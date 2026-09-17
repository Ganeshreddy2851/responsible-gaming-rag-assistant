from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


def get_retriever(k: int = TOP_K):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    print("Collection count:", vector_store._collection.count())

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever

def get_mmr_retriever(k: int = TOP_K):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10
        }
    )

    return retriever