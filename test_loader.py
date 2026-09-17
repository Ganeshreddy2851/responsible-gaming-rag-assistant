from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/ncpg.pdf")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print("Original page documents:", len(documents))
print("Chunks after splitting:", len(chunks))

print("Number of pages:", len(documents))

print("\nFIRST PAGE TEXT:")
print(documents[0].page_content[:500])

print("\nFIRST PAGE METADATA:")
print(documents[0].metadata)