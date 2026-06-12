import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

st.title("Enterprise RAG Assistant - Version 2")

st.write(
    "Upload a PDF and ask questions. This version uses ChromaDB retrieval and Ollama Llama 3.2 to generate AI answers."
)

llm = ChatOllama(model="llama3.2")

def generate_answer(question, context):
    prompt = f"""
You are an enterprise document assistant.

Answer the user's question using only the context below.
If the answer is not in the context, say: "I could not find that information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content

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

        with st.spinner("Generating answer with Llama 3.2..."):
            answer = generate_answer(question, context)

        st.subheader("AI Generated Answer")
        st.write(answer)

        with st.expander("View Retrieved Context"):
            st.write(context)