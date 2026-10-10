
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG App")
st.write("Ask questions across multiple PDFs and see the source file and page.")

if not API_KEY:
    st.error("OPENAI_API_KEY not found. Check your .env file.")
    st.stop()

# Keep the index and its details between Streamlit reruns.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

uploaded_files = st.file_uploader(
    "Upload multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

chunk_size = st.number_input(
    "Chunk size", min_value=300, max_value=2000, value=1000, step=100
)
chunk_overlap = st.number_input(
    "Chunk overlap", min_value=0, max_value=500, value=150, step=50
)

if st.button("Build / Rebuild Index", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        try:
            documents = []

            with st.spinner("Reading PDFs..."):
                for pdf in uploaded_files:
                    reader = PdfReader(BytesIO(pdf.getvalue()))

                    for page_number, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": pdf.name,
                                        "page": page_number,
                                    },
                                )
                            )

            if not documents:
                st.error(
                    "No readable PDF text found. Scanned PDFs may require OCR."
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
                chunks = splitter.split_documents(documents)

                with st.spinner("Creating embeddings and FAISS index..."):
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        api_key=API_KEY,
                    )
                    store = FAISS.from_documents(chunks, embeddings)

                # Save the single combined vector database in session state.
                st.session_state.vector_store = store
                st.session_state.chunk_count = len(chunks)
                st.session_state.indexed_files = [
                    pdf.name for pdf in uploaded_files
                ]

                st.success("Index built successfully!")
        except Exception as error:
            st.error(f"Indexing failed: {error}")

if st.session_state.vector_store is not None:
    st.subheader("Index Status")
    st.metric("Total chunks indexed", st.session_state.chunk_count)
    st.write("PDFs indexed:", ", ".join(st.session_state.indexed_files))

    question = st.text_input(
        "Ask a question",
        placeholder="Example: Summarize the main findings across all PDFs",
    )
    top_k = st.slider("Number of chunks to retrieve", 2, 8, 4)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    retrieved_docs = (
                        st.session_state.vector_store.similarity_search(
                            question, k=top_k
                        )
                    )

                    context_parts = []
                    for i, doc in enumerate(retrieved_docs, start=1):
                        filename = doc.metadata.get("source", "Unknown file")
                        page = doc.metadata.get("page", "Unknown")
                        context_parts.append(
                            f"[Source {i}: {filename}, page {page}]\n"
                            f"{doc.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        api_key=API_KEY,
                    )

                    response = llm.invoke([
                        (
                            "system",
                            "Answer using only the supplied PDF excerpts. "
                            "If the excerpts do not answer the question, say so. "
                            "Do not invent facts. Cite supporting sources inline "
                            "as [filename, page N].",
                        ),
                        (
                            "human",
                            f"Question:\n{question}\n\n"
                            f"Retrieved PDF excerpts:\n{context}",
                        ),
                    ])

                st.subheader("Answer")
                st.write(response.content)

                st.subheader("Sources")
                seen = set()
                for doc in retrieved_docs:
                    filename = doc.metadata.get("source", "Unknown file")
                    page = doc.metadata.get("page", "Unknown")
                    source = (filename, page)

                    if source not in seen:
                        seen.add(source)
                        st.markdown(f"- **File:** `{filename}` | **Page:** {page}")

            except Exception as error:
                st.error(f"Question answering failed: {error}")
else:
    st.info("Upload your PDFs and click Build / Rebuild Index to begin.")
