from langchain.text_splitter import NLTKTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta, date
import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from markdownify import markdownify as md


load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def clean_combine_text(data):
    text = ""
    for c, object in enumerate(data):
        print(c)
        text += object['text'].split('SearchSearch')[-1]
    return text

def fetch_json_by_date(directory, date_str=None): # for custom date_str format "2024-05-26" 
    if date_str is None:
        target_date = date.today()  # Set default value to today's date
    else:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    for filename in os.listdir(directory):

        if filename.endswith(".json"):

            try:
                file_date_str = filename.split("_")[2]
                file_date = datetime.strptime(file_date_str.split(' ')[0], "%Y-%m-%d").date()
                
                if file_date == target_date:
                    file_path = os.path.join(directory, filename)
                    loader = JSONLoader(
                                file_path=file_path,
                                jq_schema=".[] | .text",
                                text_content=False
                                )
                    return loader.load()
                    
            except (IndexError, ValueError):
                continue

    return None

def load_pdfs_from_directory(directory_path):
    pdf_texts = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            
            # Open and read the PDF file
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                pdf_texts.append((filename, text))
    
    return pdf_texts

def convert_pdfs_to_md(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(output_directory, md_filename)
            
            with open(pdf_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Convert the extracted text to markdown format
            md_content = md(text)
            
            # Write the markdown content to a .md file
            with open(md_path, "w") as md_file:
                md_file.write(md_content)
            print(f"Converted {filename} to {md_filename}")

def json_directory_loader(dir_path):
    loader = DirectoryLoader(
    dir_path, 
    glob="**/*.json", 
    loader_cls=JSONLoader,
    loader_kwargs={'jq_schema': '.[] | .text'}
    )
    documents = loader.load()
    return documents

def load_pdfs_from_directory(directory_path):
    # Create a DirectoryLoader with PyPDFLoader to load all PDFs
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",  # Adjust the glob pattern to match all PDF files
        loader_cls=PyPDFLoader  # Use PyPDFLoader for loading PDF files
    )
    
    # Load all documents (PDFs) from the directory
    documents = loader.load()
    return documents

def load_md_as_docs(md_directory):
    documents = []
    
    for filename in os.listdir(md_directory):
        if filename.endswith(".md"):
            md_path = os.path.join(md_directory, filename)
            loader = UnstructuredMarkdownLoader(md_path)
            documents.extend(loader.load())
    
    return documents

def NLTKchunking(text, chunk_size):
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size)
    chunks = text_splitter.split_text(text)
    print('created', len(chunks), 'number of chunks', 'text lenght is', len(text))
    return chunks

def recursivecharacterchunking(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)
    print('created', len(chunks), 'number of chunks', 'docs lenght is', len(docs))
    return chunks

# def save_SentenceTransformer_embeddings_to_vectorstore(chunks):
#     embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store_path = "SentenceTransformer_chroma_db"

#     try:
#         # Check if the vector store file exists
#         if os.path.isdir(vector_store_path):
#             # Load existing vector store
#             #vector_store = FAISS.load_local(vector_store_path, embeddings_model, allow_dangerous_deserialization=True)
#             #print('Loaded existing vector store with', vector_store.index.ntotal, 'records')
#             vector_store = Chroma(persist_directory="./SentenceTransformer_chroma_db", embedding_function=embeddings_model)
#             print('Loaded existing vector store with', vector_store._collection.count(), 'records')
#             vector_store.add_documents(chunks)
#             print('New docs added in vectore store', vector_store._collection.count(), 'records')
#         else:
#             # If vector store file does not exist, create a new one
#             #vector_store = FAISS.from_documents(chunks, embedding=embeddings_model)
#             #vector_store.save_local(vector_store_path)
#             print('Vector store not found, creating new vector store.......')
#             vector_store = Chroma.from_documents(chunks, embeddings_model, persist_directory="./SentenceTransformer_chroma_db")
#             print('created new vector store with', vector_store._collection.count(), 'records')
    
#     except Exception as e:
#         print(f"An error occurred: {e}")

def save_google_embeddings_to_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store_path = "google_chroma_db"
    f = 50
    try:
        # Check if the vector store file exists
        if os.path.isdir(vector_store_path):
            for i in range(0, len(chunks), 50):
                chunk_size = chunks[i:f]
                vector_store = Chroma.add_documents(persist_directory="./google_chroma_db", embedding=embeddings)
                print('Loaded existing vector store with', vector_store._collection.count(), 'records')
                vector_store.add_documents(chunk_size)
                print('New docs added in vectore store', vector_store._collection.count(), 'records')
                f += 50

        else:
            print('Vector store not found, creating new vector store.......')
            for i in range(0, len(chunks), 50):
                chunk_size = chunks[i:f]
                print('chunk size', len(chunk_size))
                vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./google_chroma_db")
                print('created new vector store with', vector_store._collection.count(), 'records')
                f += 50

    except Exception as e:
        print(f"An error occurred: {e}")

def save_chatgpt_embeddings_to_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    vector_store_path = "chatgpt_chroma_db"

    try:
        # Check if the vector store file exists
        if os.path.isdir(vector_store_path):
            # Load existing vector store
            #vector_store = FAISS.load_local(vector_store_path, embeddings_model, allow_dangerous_deserialization=True)
            #print('Loaded existing vector store with', vector_store.index.ntotal, 'records')
            vector_store = Chroma(persist_directory="./chatgpt_chroma_db", embedding_function=embeddings)
            print('Loaded existing vector store with', vector_store._collection.count(), 'records')
            vector_store.add_documents(chunks)
            print('New docs added in vectore store', vector_store._collection.count(), 'records')
        else:
            # If vector store file does not exist, create a new one
            #vector_store = FAISS.from_documents(chunks, embedding=embeddings_model)
            #vector_store.save_local(vector_store_path)
            print('Vector store not found, creating new vector store.......')
            vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chatgpt_chroma_db")
            print('created new vector store with', vector_store._collection.count(), 'records')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    #docs = fetch_json_by_date('../data/dawn_articles', "2024-05-31") # for custom use parameter date_str="2024-05-26" 
    #text = clean_combine_text(data)


    # # Directory paths
    # pdf_directory = "data/pdfs"  # Replace with your directory containing PDFs
    # md_directory = "data/mk"  # Directory to save converted markdown files

    # # Convert PDFs to Markdown
    # convert_pdfs_to_md(pdf_directory, md_directory)

    # # Load Markdown files as documents using UnstructuredMarkdownLoader
    # docs = load_md_as_docs(md_directory)
    docs = []
    pdfs = ["data/pdfs/Boundless_Technologies_Profile.pdf", "data/pdfs/Prowess_Presentation.pdf"]

    for pdf in pdfs:
        loader = PyPDFLoader(pdf, extract_images=True)
        docs.extend(loader.load())
    

    print(docs)

    if docs:
        chunks = recursivecharacterchunking(docs, 500, 100)
        #save_SentenceTransformer_embeddings_to_vectorstore(chunks)
        #save_google_embeddings_to_vectorstore(chunks)
        save_chatgpt_embeddings_to_vectorstore(chunks)
    else:
        print('json is not present on specified date')
    

