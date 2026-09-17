from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_app.chains import build_rag_chain
from rag_app.config import LLM_MODEL

rag_chain = build_rag_chain(k=3)

judge_llm = ChatOpenAI(
    model=LLM_MODEL
)

relevance_prompt = ChatPromptTemplate.from_template("""
You are evaluating the relevance of a RAG answer.

Determine whether the answer directly and sufficiently
addresses the user's question.

Important rules:

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

relevance_chain = (
    relevance_prompt
    | judge_llm
    | StrOutputParser()
)

test_questions = [
    "What does the document say about self-exclusion?",
    "What protections can players use to control how much they spend or how long they play?",
    "Can kids gamble?",
    "How should gambling be advertised?",
    "What options are available if a player wants to stop themselves from gambling for a long period?",
    "What is the CEO's favorite food?"
]

relevant_count = 0


for question in test_questions:

    result = rag_chain.invoke(question)

    answer = result["answer"]
    context = result["context"]

    judgment = relevance_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    }).strip()

    if judgment == "RELEVANT":
        relevant_count += 1

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nJUDGMENT:")
    print(judgment)


total = len(test_questions)

relevance_percentage = (
    relevant_count / total
) * 100


print("\n" + "=" * 70)
print("ANSWER RELEVANCE RESULTS")
print("=" * 70)

print(
    f"Relevant: {relevant_count}/{total} "
    f"({relevance_percentage:.1f}%)"
)


