import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.title("Enterprise RAG Assistant")

st.write(
    "Upload a PDF and ask questions. This deployable version uses ChromaDB and HuggingFace embeddings for semantic document search."
)


def generate_answer(question, context):
    return f"""
Based on the uploaded PDF, here is the most relevant information:

{context}

Question asked:
{question}
"""


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.subheader("Number of Chunks")
    st.write(len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    st.success("Chunks stored in ChromaDB successfully!")

    question = st.text_input("Ask a question about your PDF")

    if question:
        results = vector_store.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        st.subheader("Retrieved Answer Context")
        st.write(generate_answer(question, context))

        with st.expander("View Retrieved Context"):
            st.write(context)