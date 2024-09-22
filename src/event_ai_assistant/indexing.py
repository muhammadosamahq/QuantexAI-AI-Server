import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()


def load_pdfs_from_directory(directory: str) -> List:
    docs = []
    
    # Get all PDF files from the directory
    pdfs = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]
    
    # Loop through each PDF file and load the content
    for pdf in pdfs:
        loader = PyPDFLoader(pdf, extract_images=True)  # Assuming extract_images is an argument for PyPDFLoader
        docs.extend(loader.load())
    
    return docs

def embed_text(docs, chunk_size: int, chunk_overlap: int):
    print("Embedding")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)
    print('created', len(chunks), 'number of chunks', 'docs lenght is', len(docs))
    return chunks

def save_embeddings_to_db(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    vector_store_path = "chatgpt_chroma_db"

    try:
        print('Creating vector store.......')
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
        print('created new vector store with', vector_store._collection.count(), 'records')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    directory_path = "data/pdfs"
    docs = load_pdfs_from_directory(directory_path)

    if docs:
        chunks = embed_text(docs, 500, 100)
        #save_SentenceTransformer_embeddings_to_vectorstore(chunks)
        #save_google_embeddings_to_vectorstore(chunks)
        save_embeddings_to_db(chunks)
    else:
        print('json is not present on specified date')