import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_app.config import LLM_MODEL
from rag_app.retrieval import get_retriever

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    test_cases = json.load(file)


retriever = get_retriever(k=3)

llm = ChatOpenAI(
    model=LLM_MODEL
)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question into a clear search query
for retrieving relevant information from responsible
gambling documents.

Preserve the original meaning.

Use relevant responsible-gambling terminology when it
helps express the user's intent.

Do not answer the question.
Return only the rewritten search query.

Question:
{question}
""")

rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)

original_hits = 0
rewritten_hits = 0


for test in test_cases:

    question = test["question"]
    expected_pages = test["expected_pages"]

    # Original retrieval
    original_docs = retriever.invoke(question)

    original_pages = [
        doc.metadata.get("page", 0) + 1
        for doc in original_docs
    ]

    original_hit = any(
        page in expected_pages
        for page in original_pages
    )

    # Rewrite the question
    rewritten_question = rewrite_chain.invoke({
        "question": question
    }).strip()

    # Retrieval using rewritten question
    rewritten_docs = retriever.invoke(
        rewritten_question
    )

    rewritten_pages = [
        doc.metadata.get("page", 0) + 1
        for doc in rewritten_docs
    ]

    rewritten_hit = any(
        page in expected_pages
        for page in rewritten_pages
    )

    if original_hit:
        original_hits += 1

    if rewritten_hit:
        rewritten_hits += 1

    print("\n" + "=" * 60)

    print("ORIGINAL QUESTION:")
    print(question)

    print("\nREWRITTEN QUESTION:")
    print(rewritten_question)

    print(
        f"\nOriginal:  {original_pages} | "
        f"Hit@3: {original_hit}"
    )

    print(
        f"Rewritten: {rewritten_pages} | "
        f"Hit@3: {rewritten_hit}"
    )


total = len(test_cases)

print("\n" + "=" * 60)
print("QUERY REWRITING COMPARISON")
print("=" * 60)

print(
    f"Original Hit@3: "
    f"{original_hits}/{total} "
    f"({original_hits / total * 100:.1f}%)"
)

print(
    f"Rewritten Hit@3: "
    f"{rewritten_hits}/{total} "
    f"({rewritten_hits / total * 100:.1f}%)"
)