from rag_app.chains import build_rag_chain

rag_chain = build_rag_chain(k=3)


test_questions = [
    "What does the document say about self-exclusion?",

    "What protections can players use to control how much they spend or how long they play?",

    "Can kids gamble?",

    "How should gambling be advertised?",

    "What options are available if a player wants to stop themselves from gambling for a long period?",

    # Intentionally unsupported question
    "What is the CEO's favorite food?"
]


for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    result = rag_chain.invoke(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")

    for doc in result["docs"]:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        print(
            f"- {source} | Page {page}"
        )