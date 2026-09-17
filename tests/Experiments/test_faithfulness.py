from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_app.chains import build_rag_chain
from rag_app.config import LLM_MODEL

rag_chain = build_rag_chain(k=3)

judge_llm = ChatOpenAI(model=LLM_MODEL)

judge_prompt = ChatPromptTemplate.from_template("""
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

judge_chain = judge_prompt | judge_llm | StrOutputParser()

test_questions = [
    "What does the document say about self-exclusion?",
    "What protections can players use to control how much they spend or how long they play?",
    "Can kids gamble?",
    "How should gambling be advertised?",
    "What options are available if a player wants to stop themselves from gambling for a long period?",
    "What is the CEO's favorite food?",
]

faithful_count = 0


for question in test_questions:
    result = rag_chain.invoke(question)

    answer = result["answer"]
    context = result["context"]

    judgment = judge_chain.invoke(
        {"question": question, "context": context, "answer": answer}
    ).strip()

    if judgment == "FAITHFUL":
        faithful_count += 1

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nJUDGMENT:")
    print(judgment)

total = len(test_questions)

faithfulness_percentage = (faithful_count / total) * 100


print("\n" + "=" * 70)
print("FAITHFULNESS RESULTS")
print("=" * 70)

print(f"Faithful: {faithful_count}/{total} ({faithfulness_percentage:.1f}%)")
