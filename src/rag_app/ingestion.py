from pathlib import Path
import hashlib # If there is any update in existing document(Document lifecycle management)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def get_file_hash(file_path: Path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_vector_store(pdf_path: str):
    pdf_path = Path(pdf_path).resolve()

    source_path = str(pdf_path)
    file_name = pdf_path.name
    file_hash = get_file_hash(pdf_path)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    existing = vector_store._collection.get(
        where={"source": source_path},
        include=["metadatas"]
    )

    # Document already exists
    if len(existing["ids"]) > 0:

        existing_hash = existing["metadatas"][0].get("file_hash")

        # Same document, same contents
        if existing_hash == file_hash:
            print(f"{file_name} is already up to date.")
            return vector_store

        # Same document path, but contents changed
        print(f"{file_name} has changed. Replacing old chunks...")

        vector_store._collection.delete(
            where={"source": source_path}
        )

    loader = PyPDFLoader(source_path)
    documents = loader.load()

    # Add our own hash metadata
    for document in documents:
        document.metadata["file_hash"] = file_hash

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    vector_store.add_documents(chunks)

    print(f"{file_name}: {len(chunks)} chunks added.")

    return vector_store

def ingest_directory(directory_path: str):
    directory = Path(directory_path)

    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF file(s).\n")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        vector_store = build_vector_store(str(pdf_file))

        print("-" * 50)

    print("\nDirectory ingestion complete.")

    return vector_store