from rag_app.chains import build_rag_chain

rag_chain = build_rag_chain(k=3)

print("RAG Assistant")
print("Type 'exit' to quit.\n")

while True:
    question = input("Question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    result = rag_chain.invoke(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    seen_sources = set()

    for doc in result["docs"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1

        source_info = (source, page)

        if source_info not in seen_sources:
            print(f"- {source} — Page {page}")
            seen_sources.add(source_info)