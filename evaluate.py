import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_app.chains import build_rag_chain
from rag_app.config import LLM_MODEL
from rag_app.retrieval import get_retriever

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    retrieval_tests = json.load(file)


with open(
    "tests/answer_evaluation.json",
    "r",
    encoding="utf-8"
) as file:
    answer_tests = json.load(file)

retriever_k1 = get_retriever(k=1)
retriever_k3 = get_retriever(k=3)
retriever_k5 = get_retriever(k=5)

rag_chain = build_rag_chain(k=3)

judge_llm = ChatOpenAI(
    model=LLM_MODEL
)

faithfulness_prompt = ChatPromptTemplate.from_template("""
You are evaluating the faithfulness of a RAG answer.

Determine whether the answer is supported by the retrieved context.

Rules:

1. If the answer contains factual claims, every factual claim
   must be supported by the retrieved context.

2. If the answer says that the information was not found in
   the provided documents, consider this FAITHFUL when the
   retrieved context does not contain enough information to
   answer the question.

3. An abstention is NOT_FAITHFUL if the retrieved context
   clearly contains enough information to answer the question.

Return only one label:

FAITHFUL
NOT_FAITHFUL

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}
""")


relevance_prompt = ChatPromptTemplate.from_template("""
You are evaluating the relevance of a RAG answer.

Determine whether the answer directly and sufficiently
addresses the user's question.

Rules:

1. Judge whether the answer addresses what the user
   actually asked.

2. An answer can contain true information but still be
   NOT_RELEVANT if it does not answer the question.

3. If the answer says the information was not found,
   return RELEVANT only when the retrieved context genuinely
   does not contain enough information to answer the question.

4. If relevant information exists in the retrieved context
   but the answer ignores it, return NOT_RELEVANT.

Return only one label:

RELEVANT
NOT_RELEVANT

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}
""")

faithfulness_chain = (
    faithfulness_prompt
    | judge_llm
    | StrOutputParser()
)


relevance_chain = (
    relevance_prompt
    | judge_llm
    | StrOutputParser()
)

def evaluate_retrieval():

    hit1 = 0
    hit3 = 0
    hit5 = 0

    total = len(retrieval_tests)

    for test in retrieval_tests:

        question = test["question"]
        expected_pages = test["expected_pages"]

        docs_k1 = retriever_k1.invoke(question)
        docs_k3 = retriever_k3.invoke(question)
        docs_k5 = retriever_k5.invoke(question)

        pages_k1 = [
            doc.metadata.get("page", 0) + 1
            for doc in docs_k1
        ]

        pages_k3 = [
            doc.metadata.get("page", 0) + 1
            for doc in docs_k3
        ]

        pages_k5 = [
            doc.metadata.get("page", 0) + 1
            for doc in docs_k5
        ]

        if any(
            page in expected_pages
            for page in pages_k1
        ):
            hit1 += 1

        if any(
            page in expected_pages
            for page in pages_k3
        ):
            hit3 += 1

        if any(
            page in expected_pages
            for page in pages_k5
        ):
            hit5 += 1

    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total
    }

def evaluate_answers():

    passed = 0
    abstention_passed = 0
    abstention_total = 0

    total = len(answer_tests)

    abstention_text = (
        "The information was not found in the provided documents."
    )

    for test in answer_tests:

        question = test["question"]
        expected_keywords = test["expected_keywords"]
        should_abstain = test["should_abstain"]

        result = rag_chain.invoke(question)

        answer = result["answer"]

        if should_abstain:

            abstention_total += 1

            test_passed = (
                abstention_text.lower()
                in answer.lower()
            )

            if test_passed:
                abstention_passed += 1

        else:

            test_passed = all(
                keyword.lower() in answer.lower()
                for keyword in expected_keywords
            )

        if test_passed:
            passed += 1

        if not test_passed:
            print("\nFAILED ANSWER TEST")
            print("Question:", question)
            print("Expected keywords:", expected_keywords)
            print("Answer:", answer)

    return {
        "answer_pass_rate": passed / total,
        "abstention_pass_rate": (
            abstention_passed / abstention_total
            if abstention_total > 0
            else 0
        )
    }


def evaluate_judges():

    test_questions = [
        "What does the document say about self-exclusion?",
        "What protections can players use to control how much they spend or how long they play?",
        "Can kids gamble?",
        "How should gambling be advertised?",
        "What options are available if a player wants to stop themselves from gambling for a long period?",
        "What is the CEO's favorite food?"
    ]

    faithful_count = 0
    relevant_count = 0

    for question in test_questions:

        result = rag_chain.invoke(question)

        answer = result["answer"]
        context = result["context"]

        faithfulness = faithfulness_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        }).strip()

        relevance = relevance_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        }).strip()

        if faithfulness == "FAITHFUL":
            faithful_count += 1

        if relevance == "RELEVANT":
            relevant_count += 1

        if (
            faithfulness != "FAITHFUL"
            or relevance != "RELEVANT"
        ):
            print("\nJUDGE FAILURE")
            print("Question:", question)
            print("Answer:", answer)
            print("Faithfulness:", faithfulness)
            print("Relevance:", relevance)

    total = len(test_questions)

    return {
        "faithfulness": faithful_count / total,
        "answer_relevance": relevant_count / total
    }


retrieval_results = evaluate_retrieval()
answer_results = evaluate_answers()
judge_results = evaluate_judges()


print("\n" + "=" * 60)
print("RAG EVALUATION REPORT")
print("=" * 60)

print("\nRETRIEVAL")
print(
    f"Hit@1: "
    f"{retrieval_results['hit@1'] * 100:.1f}%"
)
print(
    f"Hit@3: "
    f"{retrieval_results['hit@3'] * 100:.1f}%"
)
print(
    f"Hit@5: "
    f"{retrieval_results['hit@5'] * 100:.1f}%"
)

print("\nGENERATION")
print(
    f"Answer pass rate: "
    f"{answer_results['answer_pass_rate'] * 100:.1f}%"
)
print(
    f"Abstention pass rate: "
    f"{answer_results['abstention_pass_rate'] * 100:.1f}%"
)
print(
    f"Faithfulness: "
    f"{judge_results['faithfulness'] * 100:.1f}%"
)
print(
    f"Answer relevance: "
    f"{judge_results['answer_relevance'] * 100:.1f}%"
)

print("\n" + "=" * 60)