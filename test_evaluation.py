import json

from rag_app.retrieval import get_mmr_retriever, get_retriever

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    test_cases = json.load(file)

k_values = [1, 3, 5]

results = {k: 0 for k in k_values}


for test in test_cases:

    question = test["question"]
    expected_pages = test["expected_pages"]

    print("\nQuestion:")
    print(question)

    print("Expected pages:")
    print(expected_pages)

    for k in k_values:

        retriever = get_retriever(k=k)

        docs = retriever.invoke(question)

        if k == 5:
            print("\nTOP 5 RETRIEVED CHUNKS:")

            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", 0) + 1
                print(f"\n--- Result {i} | PDF Page {page} ---")
                print(doc.page_content[:500])

        retrieved_pages = [
            doc.metadata.get("page", 0) + 1
            for doc in docs
        ]

        hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if hit:
            results[k] += 1

        print(
            f"Hit@{k}: {hit} | "
            f"Retrieved pages: {retrieved_pages}"
        )

    print("-" * 60)


print("\nFINAL RESULTS")

total_questions = len(test_cases)

for k in k_values:

    score = results[k]
    percentage = (score / total_questions) * 100

    print(
        f"Hit@{k}: "
        f"{score}/{total_questions} "
        f"({percentage:.1f}%)"
    )

print("\n" + "=" * 60)
print("SIMILARITY vs MMR — Hit@3")
print("=" * 60)

similarity_retriever = get_retriever(k=3)
mmr_retriever = get_mmr_retriever(k=3)

similarity_hits = 0
mmr_hits = 0

for test in test_cases:

    question = test["question"]
    expected_pages = test["expected_pages"]

    similarity_docs = similarity_retriever.invoke(question)
    mmr_docs = mmr_retriever.invoke(question)

    similarity_pages = [
        doc.metadata.get("page", 0) + 1
        for doc in similarity_docs
    ]

    mmr_pages = [
        doc.metadata.get("page", 0) + 1
        for doc in mmr_docs
    ]

    similarity_hit = any(
        page in expected_pages
        for page in similarity_pages
    )

    mmr_hit = any(
        page in expected_pages
        for page in mmr_pages
    )

    if similarity_hit:
        similarity_hits += 1

    if mmr_hit:
        mmr_hits += 1

    print(f"\nQuestion: {question}")
    print(f"Similarity: {similarity_pages} | Hit: {similarity_hit}")
    print(f"MMR:        {mmr_pages} | Hit: {mmr_hit}")


total = len(test_cases)

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

print(
    f"Similarity Hit@3: "
    f"{similarity_hits}/{total} "
    f"({similarity_hits / total * 100:.1f}%)"
)

print(
    f"MMR Hit@3: "
    f"{mmr_hits}/{total} "
    f"({mmr_hits / total * 100:.1f}%)"
)