import streamlit as st 
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    model = SentenceTransformer(
        "dunzhang/stella_en_400M_v5",
        trust_remote_code=True,
        device="cpu",
        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
    )
    doc_embeddings = model.encode(text_chunks)
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=doc_embeddings)
    

    return vectorstore
    



def main():
    load_dotenv()
    # Design page layout 
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask a question about your documents: ")

    # Side bar 
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click 'Process' ", accept_multiple_files=True)
        if st.button("Process"):
            # add streamlit loading effect
            with st.spinner("Processing"):
                # Get pdf texts 
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks 
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # Create vector store with the embeddings 
                vectorstore = get_vectorstore(text_chunks)
                
                if vectorstore:
                    st.success("PDF processed successfully!")
                else:
                    st.error("Failed to process PDF")


if __name__ == "__main__":
    main()