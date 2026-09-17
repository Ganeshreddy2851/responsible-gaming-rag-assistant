from rag_app.ingestion import build_vector_store

vector_store = build_vector_store("data/ncpg.pdf")

print("Stored chunks:", vector_store._collection.count())