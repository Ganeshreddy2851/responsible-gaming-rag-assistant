from rag_app.config import TOP_K, LLM_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag_app.retrieval import get_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(k: int = TOP_K):
    retriever = get_retriever(k=k)

    prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the context below.

If the answer cannot be found in the context, say:
"The information was not found in the provided documents."

Context:
{context}

Question:
{question}
""")

    llm = ChatOpenAI(model=LLM_MODEL)

    rag_chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(
            answer=(
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    return rag_chain


def build_conversational_rag_chain(k=TOP_K):
    retriever = get_retriever(k=k)

    llm = ChatOpenAI(model=LLM_MODEL)

    rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the user's latest question,
rewrite the latest question as a complete standalone question.

The rewritten question must make sense without seeing the
conversation history.

If the question is already standalone, return it unchanged.

Do not answer the question.
Only return the rewritten question.

Conversation history:
{chat_history}

Latest question:
{question}
""")

    answer_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the context below.

If the answer cannot be found in the context, say:
"The information was not found in the provided documents."

Context:
{context}

Question:
{question}
""")

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=(
                {
                    "chat_history": lambda x: x["chat_history"],
                    "question": lambda x: x["question"],
                }
                | rewrite_chain
            )
        )
        | RunnablePassthrough.assign(
            docs=lambda x: retriever.invoke(x["standalone_question"])
        )
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(
            answer=(
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | answer_prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    return rag_chain
