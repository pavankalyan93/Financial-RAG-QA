import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# --- Load Environment Variables ---
# Make sure to create a .env file with your OPENAI_API_KEY
load_dotenv()

# --- App Configuration ---
st.set_page_config(page_title="Financial Document Q&A", layout="wide")
st.title("Financial Document Q&A with RAG 🤖")
st.markdown("---")

# --- Helper Functions with Caching ---
@st.cache_resource
def load_and_process_document(file_path):
    """
    Loads a text document, splits it into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    st.info(f"Processing document: {file_path}...")
    # 1. Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    # 3. Create embeddings using a HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Create a FAISS vector store from the chunks
    vector_store = FAISS.from_documents(texts, embeddings)
    st.success("Document processed and vector store created successfully!")
    return vector_store

@st.cache_data
def get_qa_chain(_vector_store):
    """
    Initializes and returns a RetrievalQA chain.
    """
    # Initialize the LLM (Large Language Model)
    llm = OpenAI(temperature=0.1)
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever()
    )
    return qa_chain

# --- Main Application Logic ---

# 1. Create a dummy document for the demonstration
# This simulates having a financial report ready for analysis.
dummy_text = """
Annual Report for Innovate Corp - Fiscal Year 2024

CEO Statement:
This has been a landmark year for Innovate Corp. We achieved a record revenue of $1.2 billion, a 15% increase from the previous year. Our net profit stood at $150 million. The primary driver for this growth was our new AI-powered analytics platform, "Synapse," which accounted for 40% of new sales.

Financial Highlights:
- Total Revenue: $1.2 billion
- Net Profit: $150 million
- R&D Spending: $200 million, a 25% increase, focusing on generative AI.

Future Outlook:
We plan to expand our market presence in Europe and increase our R&D budget by another 30% in 2025. Our new product, "Cognito," a generative AI copilot, is scheduled for a Q3 2025 launch.
"""
DEMO_FILE_PATH = "annual_report_2024.txt"
with open(DEMO_FILE_PATH, "w") as f:
    f.write(dummy_text)

st.success(f"Loaded '{DEMO_FILE_PATH}' for demonstration.")
st.markdown("---")

# 2. Load the document and initialize the QA chain
# This only runs once thanks to caching.
try:
    if os.environ.get("OPENAI_API_KEY"):
        vector_store = load_and_process_document(DEMO_FILE_PATH)
        qa_chain = get_qa_chain(vector_store)

        # 3. Build the User Interface
        st.header("Ask a Question About the Document")
        user_question = st.text_input("Enter your question below:")

        if user_question:
            with st.spinner("Generating answer..."):
                try:
                    answer = qa_chain.run(user_question)
                    st.markdown("### Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
    else:
        st.error("OPENAI_API_KEY not found. Please create a .env file with your key.")

except Exception as e:
    st.error(f"An error occurred during setup: {e}")
