from pathlib import Path
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "updated_file_with_lat_long.csv"

def initialize_chain():
    # Load documents
    loader = CSVLoader(csv_path, encoding="utf-8")
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    # Initialize embeddings and vector store
    hugemb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.from_documents(documents, hugemb)

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        input_key="question"
    )

    # Initialize LLM
    groqllm = ChatGroq(
        groq_api_key=groq_api_key,
        model='llama3-8b-8192'
    )

    # Create chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=groqllm,
        retriever=db.as_retriever(),
        memory=memory
    )

    return conversation_chain, memory

# Lazy init to avoid loading on module import
conversation_chain, memory = initialize_chain()

class ChatbotView(APIView):
    def post(self, request):
        user_input = request.data.get("question", "")

        if not user_input:
            return Response(
                {"error": "Question is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            response = conversation_chain.invoke({
                'question': user_input,
                'chat_history': memory.buffer
            })
            return Response({"answer": response['answer']})
        except Exception as e:
            return Response(
                {"error": f"Internal Server Error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
