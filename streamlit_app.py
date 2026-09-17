import streamlit as st

from rag_app.chains import build_conversational_rag_chain

st.set_page_config(
    page_title="Responsible Gaming RAG Assistant",
    page_icon="🎯"
)


@st.cache_resource
def load_rag_chain():
    return build_conversational_rag_chain()


rag_chain = load_rag_chain()


st.title("Responsible Gaming RAG Assistant")

st.write(
    "Ask questions about the responsible gaming documents "
    "stored in the knowledge base."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about responsible gaming..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            chat_history = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in st.session_state.messages[:-1]
            )

            result = rag_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
            })

        answer = result["answer"]

        st.markdown(answer)

        with st.expander("Rewritten retrieval question"):
            st.write(result["standalone_question"])
        with st.expander("Retrieved context"):
            for i, doc in enumerate(result["docs"], start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0) + 1

                st.caption(
                f"Source: {source} | Page: {page}"
                )

                st.divider()   
        with st.expander("Sources"):

            seen_sources = set()

            for doc in result["docs"]:

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0) + 1

                source_info = (source, page)

                if source_info not in seen_sources:

                    st.write(
                        f"{source} — Page {page}"
                    )

                    seen_sources.add(source_info)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })