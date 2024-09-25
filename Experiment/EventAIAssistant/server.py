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
from langchain_openai import OpenAIEmbeddings
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from prompt_template import prompt_template
#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv


import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os


from langchain.chains import create_history_aware_retriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


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

class LanguageModelProcessor:
    def __init__(self):
        #self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = Chroma(persist_directory="./chatgpt_chroma_db", embedding_function=self.embeddings).as_retriever()

        self.system_prompt = """Role: You are an intelligent, context-aware bot designed to assist attendees of a specific event. Your purpose is to provide accurate, concise information regarding the event, the company hosting it, and the company’s products. If exact information is unavailable, you should try to offer nearby or related information. If a query is highly irrelevant or unclear, politely ask for more context to assist the attendee properly.

        Instructions:

        Contextual Knowledge Source: You have access to a collection of documents containing all event-related information, the company’s history, details about the event (dates, times, location, agenda), and product details of the hosting company.

        Task Execution:

        If the attendee's query pertains to:
        Event Information: Respond with concise details about event timing, location, agenda, speakers, and any other relevant information.
        Company Information: Provide short, key details about the hosting company, its history, mission, or organizational structure.
        Product Information: Offer brief details about the company’s products or services, highlighting features, use cases, and pricing when relevant.
        Nearby Information Rule:

        If you cannot find an exact answer but can infer related information from the context, provide that related information, making it clear that it's approximate.
        Example:
        "I don’t have the exact answer to your question, but based on similar details, I can offer this information..."
        Ask for More Context:

        If the attendee’s query is highly irrelevant or unclear, ask for more context to assist effectively. Respond as follows:
        “I couldn’t fully understand your question. Could you please provide me a bit more context? This will help me give you the exact answer you need.”
        Short Answers First:

        Whenever possible, provide a short, direct answer rather than asking for more context, unless the query is clearly irrelevant or vague.
        Tone: Use a professional and helpful tone. Keep responses concise but informative.

        Restrictions: Do not provide speculative answers. If no relevant information is found, ask for clarification or more context rather than providing an unsupported answer.
           
        {context}"""

        self.prompt_with_history = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_with_history)

        self.contextualize_prompt  = ChatPromptTemplate.from_messages(
                    [
                        ("system", 
                        "Given a chat history and the latest user question which might reference context in the chat history,"
                            "formulate a standalone question which can be understood without the chat history."
                            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
                            ),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),


                    ]
                )
        
        self.history_aware_retriever = create_history_aware_retriever(self.llm ,self.retriever ,self.contextualize_prompt)
     
        
        self.rag_chain_with_history = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain) 
        self.rag_chain = create_retrieval_chain(self.retriever, create_stuff_documents_chain(self.llm, self.prompt)) # this is used for without history

        self.store = {}

        self.conversation = RunnableWithMessageHistory(
            self.rag_chain_with_history,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        self.q_and_a_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.prompt)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def process_with_history(self, text, session_id):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        response = self.conversation.invoke(
                {"input": text},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key "abc123" in `store`.
            )
        self.memory.chat_memory.add_ai_message(response["answer"])  # Add AI response to memory
        print("LLM: ", response["answer"])
        return response["answer"]

    def process_q_and_a(self, text):
        self.docs = Chroma(persist_directory="./chatgpt_chroma_db", embedding_function=self.embeddings).similarity_search(text)
        response = self.q_and_a_chain({"input_documents": self.docs, "input": text})
        print("LLM: ", response["output_text"])
        return response["output_text"]
    
    def process(self, text):
        response = self.rag_chain.invoke({"input": text})
        print("LLM: ", response["answer"])
        return response["answer"]
    
    def response_latency(self, text):
        start_time = time.time()
        response = self.process_with_history(text)
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"process_with_history latency ({elapsed_time}ms): {response}")

        start_time = time.time()
        response = self.process_q_and_a(text)
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"process_q_and_a latency ({elapsed_time}ms): {response}")

        start_time = time.time()
        response = self.process(text)
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"process latency ({elapsed_time}ms): {response}")

def get_conversational_chain():
    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, model_name="gpt-4o-mini")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def input_response(user_question: str):
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    new_db = Chroma(persist_directory="./chatgpt_chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)
    print(docs)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response
    

class QueryRequest(BaseModel):
    station_id: int
    query: str

current_face = None
intro_sent = False

@app.get("/query/")
async def get_response(station_id: int = Query(...), query: str = Query(...)):
    llm = LanguageModelProcessor()
    response = requests.get("http://localhost:5000/detect_face")
    try:
        if response.status_code == 200:
            data = response.json()
            detected_face = data.get("face_detected")

            if detected_face:
                print("Face detected!")
                # current_face = detected_face
                # intro_sent = False
                # if not intro_sent:
                intro_message = "Welcome! How can I assist you today?"
                return {"station_id": station_id, "response": intro_message}
                # llm_response = llm.process(query)
                # print(query)
                # response = llm.process_with_history(query, 123)
                # #response = input_response(query)
                # print("llm_response:", response)
                # return {"station_id": station_id, "response": response} #response["output_text"]
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
