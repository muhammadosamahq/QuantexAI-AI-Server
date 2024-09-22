import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import cv2
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.chains import create_history_aware_retriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.question_answering import load_qa_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()


class LanguageModelProcessor:
    def __init__(self):
        #self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.retriever = Chroma(persist_directory="./chatgpt_chroma_db", embedding_function=self.embeddings).as_retriever()

        self.system_prompt = """Role: You are an intelligent, context-aware bot designed to assist attendees of a specific event. Your purpose is to provide accurate information regarding the event, the company hosting it, and the company’s products. If exact information is unavailable, you should try to offer nearby or related information. If you cannot find relevant information or do not fully understand the attendee’s query, politely ask for more context to provide a better answer.

                        Instructions:

                        Contextual Knowledge Source: You have access to a collection of documents containing all event-related information, the company’s history, details about the event (dates, times, location, agenda), and product details of the hosting company.

                        Task Execution:

                        If the attendee's query pertains to:
                        Event Information: Respond with details about event timing, location, agenda, speakers, and any other relevant information.
                        Company Information: Provide key details about the hosting company, its history, mission, or organizational structure.
                        Product Information: Respond with details about the products or services the company offers, highlighting features, use cases, and any pricing if relevant.
                        Nearby Information Rule:

                        If you cannot find an exact answer but can infer related information from the context, provide that related information, making it clear that it's approximate.
                        Example:
                        "I don’t have the exact answer to your question, but based on similar details, I can offer this information..."
                        Ask for More Context:

                        If you do not have enough information to answer the question or the query is unclear, respond as follows:
                        "I couldn’t fully understand your question. Could you please provide me a bit more context? This will help me give you the exact answer you need."
                        Tone: Use a professional and helpful tone. Keep responses concise but informative.

                        Restrictions: Do not provide speculative answers. If no relevant information is found, ask for clarification or more context rather than giving an unsupported answer.
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

    def process_with_history(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        response = self.conversation.invoke(
                {"input": text},
                config={
                    "configurable": {"session_id": "123"}
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

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1")
        print ("Listening...")

        # print('''Hello and welcome to [Company Name]'s event! I'm your AI assistant, here to guide you through everything 
        #                     you need to know today. Whether you're exploring our latest products, looking for session details, or seeking personalized recommendations, 
        #                     I'm here to assist you every step of the way. Just ask, and I'll provide you with up-to-date information, event highlights, and more. 
        #                     Let's make this experience exceptional—how can I help you today''')

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print('''fillers''')
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=500,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

        

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            response = requests.get("http://localhost:8000/detect_face")
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Check if a face was detected
                if data.get("face_detected"):
                    print("Face detected!")

                    await get_transcript(handle_full_sentence)
                    
                    # Check for "goodbye" to exit the loop
                    if "goodbye" in self.transcription_response.lower():
                        break
                    
                    llm_response = self.llm.process(self.transcription_response)
                    print(llm_response)

                    # Reset transcription_response for the next loop iteration
                    self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())