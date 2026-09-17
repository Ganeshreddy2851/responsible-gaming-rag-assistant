import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
)

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    test_cases = json.load(file)


pdf_path = Path("data/ncpg.pdf").resolve()

chunk_sizes = [500, 800, 1200]
chunk_overlap = 100
k = 3

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

documents = loader.load()

results = {}


for chunk_size in chunk_sizes:

    print("\n" + "=" * 60)
    print(f"TESTING CHUNK SIZE: {chunk_size}")
    print("=" * 60)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    collection_name = f"chunk_size_{chunk_size}_experiment"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    vector_store._client.delete_collection(
        name=collection_name
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    hits = 0

    for test in test_cases:

        question = test["question"]
        expected_pages = test["expected_pages"]

        docs = retriever.invoke(question)

        retrieved_pages = [
            doc.metadata.get("page", 0) + 1
            for doc in docs
        ]

        hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if hit:
            hits += 1

        print(
            f"\nQuestion: {question}\n"
            f"Pages: {retrieved_pages} | "
            f"Hit@{k}: {hit}"
        )

    percentage = (
        hits / len(test_cases)
    ) * 100

    results[chunk_size] = {
        "hits": hits,
        "percentage": percentage,
        "chunks": len(chunks)
    }

    print("\n" + "=" * 60)
print("CHUNK SIZE COMPARISON")
print("=" * 60)

for chunk_size, result in results.items():

    print(
        f"Chunk size {chunk_size}: "
        f"{result['hits']}/{len(test_cases)} "
        f"Hit@{k} "
        f"({result['percentage']:.1f}%) | "
        f"{result['chunks']} chunks"
    )