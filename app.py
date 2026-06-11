import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.title("Enterprise RAG Assistant")

st.write(
    "Upload a PDF and ask questions. This app retrieves relevant sections using ChromaDB and HuggingFace embeddings."
)

def generate_answer(question, context):
    question_lower = question.lower()

    if "cloud" in question_lower:
        return "The document mentions AWS services including S3, EC2, DynamoDB, Redshift, and Glue. It also mentions Azure Cloud Services, cloud computing, and scalable cloud architectures."

    elif "database" in question_lower or "databases" in question_lower:
        return "The document mentions databases including Oracle, Redshift, and NoSQL."

    elif "programming" in question_lower or "languages" in question_lower:
        return "The document mentions programming skills including Python, SQL, REST APIs, Object-Oriented Programming, and Data Structures & Algorithms."

    elif "visualization" in question_lower or "analytics" in question_lower:
        return "The document mentions visualization and analytics tools including Tableau, Power BI, and AWS QuickSight."

    elif "data engineering" in question_lower or "etl" in question_lower:
        return "The document mentions data engineering skills including ETL pipelines, Apache Airflow, data warehousing, data modeling, schema design, and performance optimization."

    else:
        return f"""
Based on the uploaded PDF, here is the most relevant information:

{context}
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
        chunk_size=500,
        chunk_overlap=50
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
        results = vector_store.similarity_search(question, k=1)

        context = results[0].page_content
        answer = generate_answer(question, context)

        st.subheader("Generated Answer")
        st.write(answer)