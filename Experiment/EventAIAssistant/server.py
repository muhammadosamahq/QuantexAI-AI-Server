from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
#from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from prompt_template import prompt_template
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_conversational_chain():
    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, model_name="gpt-4o-mini")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def input_response(user_question: str):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    new_db = Chroma(persist_directory="./SentenceTransformer_chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

class QueryRequest(BaseModel):
    station_id: int
    query: str

@app.get("/query/")
async def get_response(station_id: int = Query(...), query: str = Query(...)):
    try:
        response = input_response(query)
        return {"station_id": station_id, "response": response["output_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
