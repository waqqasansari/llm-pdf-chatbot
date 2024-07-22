import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from PyPDF2 import PdfReader
import markdown
import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import time
from .models import PDFDocument
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS

class ChatbotLogic:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.index_name = "testingpinecone"

        self.llm = OpenAI(api_key=self.openai_api_key, model_name="gpt-3.5-turbo-instruct")

        print("Loading and processing PDF...")
        latest_pdf = PDFDocument.objects.latest('uploaded_at')
        pdf_file_path = latest_pdf.pdf_file.path
        pdf_reader = PdfReader(pdf_file_path)
        self.docs = ""
        for page in pdf_reader.pages:
            self.docs += page.extract_text()
        print("docs" ,self.docs)

        print("Initializing embeddings and text splitting...")
        self.embeddings = OpenAIEmbeddings()
        self.semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # self.splits = self.semantic_chunker.create_documents(self.docs)
        self.splits = self.semantic_chunker.split_text(self.docs)

        print("Setting up Pinecone index...")
        pc = Pinecone(api_key=self.pinecone_api_key)

        if self.index_name not in pc.list_indexes():
            print(f"Creating index '{self.index_name}'...")
            try:
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                print("Index creation complete.")
            except Exception as e:
                print(f"Index creation failed: {e}")

        index = pc.Index(self.index_name)
        print(index.describe_index_stats())

        print("Setting up Pinecone vector store...")
        docsearch = PineconeVectorStore.from_texts(self.splits, self.embeddings, index_name=self.index_name)
        print("Vector store setup complete.")
        print(docsearch)
        self.retriever = docsearch.as_retriever()

        print("Setting up question-answering chains...")
        self.contextualize_q_system_prompt = """Given a chat history and the latest user question \
                                                which might reference context in the chat history, formulate a standalone question \
                                                which can be understood without the chat history. Do NOT answer the question, \
                                                just reformulate it if needed and otherwise return it as is."""
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )

        self.qa_system_prompt =  """You are an assistant for question-answering tasks. \
                                    Keep the conversation natural, means if question is greetigs then you answer according to that
                                    Use the following pieces of retrieved context to answer the question. \
                                    If you don't know the answer, just say that you don't know. \
                                    Use three sentences maximum and keep the answer concise.\

                                    {context}"""
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def process_message(self, session_id: str, user_input: str) -> str:
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return response