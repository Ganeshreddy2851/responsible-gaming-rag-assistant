from rag_app.chains import build_rag_chain


rag_chain = build_rag_chain(k=3)

answer = rag_chain.invoke(
    "What does the document say about responsible gaming?"
)

print("\nANSWER:")
print(answer)