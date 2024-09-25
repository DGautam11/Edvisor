from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from model import Model
from chat_manager import ChatManager
from rag import RAG
from transformers import pipeline
from typing import List, Dict
from langchain.docstore.document import Document
import re
import chromadb
from chromadb.config import Settings
from config import Config   

class Engine:
    def __init__(self):
        self.model = Model()
        self.config = Config()
       # Create a single Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_persist_directory)

        # Initialize ChatManager and RAG with the same Chroma client
        self.chat_manager = ChatManager(self.chroma_client)
        self.rag = RAG(self.chroma_client)
        
        self._setup_llm()
    
    def _setup_llm(self):
        model, tokenizer = self.model.get_model_tokenizer()
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        self._system_prompt = """You are an AI assistant for Edvisor, a chatbot specializing in Finland Study and Visa Services. 
            Follow these rules strictly:
            1. Provide helpful and accurate information only about studying in Finland, Finnish education system, and student visas.
            2. Respond directly to the user's query or greeting without generating additional context to questions.
            3. For greetings or general inquiries, respond politely and briefly, then ask how you can assist with information about studying in Finland.
            4. If you don't know the answer, politely inform the user that you are unable to assist with that specific query.
            5. If the query is not related to studying in Finland or Finnish student visas, politely inform the user that you can only assist with these topics.
            6. Always maintain a helpful and friendly tone, but focus on providing information rather than engaging in open-ended conversation.
            Remember, your primary function is to provide information about studying in Finland and related visa processes."""


    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        print(f"Generating response for user: {user_email}, message: {user_message}")

        is_greeting = self._is_greeting(user_message)
        print(f"Is greeting: {is_greeting}")

        messages = [SystemMessage(content=self._system_prompt)]

        if not is_greeting :
            print("Using full prompt with RAG")
            memory = self._load_chat_history(chat_id, user_email)
            retrieved_docs = self.rag.query_vector_store(user_message, k=2)
            retrieved_docs_text = self._prepare_retrieved_docs(retrieved_docs)
            print(f"Retrieved docs: {retrieved_docs_text}")

            context_message = f" Use below conversation summary to maintain the context. \n Conversation Summary: {memory.buffer}"
            retrieved_docs_message = f" Use the below information to inform your response.Retrieved Information:\n{retrieved_docs_text}"

            messages.extend([
                SystemMessage(content=context_message),
                SystemMessage(content=retrieved_docs_message),
            ])

        messages.append(HumanMessage(content=user_message))

        print(f"Generated messages: {messages}")

        # Generate response
        response = self.llm.invoke(messages)
        print(f"Full response from model: {response}")

        assistant_response = response.split("Human")[-1].split("System:")[-1].strip()
        print(f"Extracted assistant response: {assistant_response}")

        # Save the new message to chat manager
        self._save_message(chat_id, user_email, user_message, assistant_response)

        return assistant_response

    def _is_greeting(self, message: str) -> bool:
        greeting_patterns = [
            r'\b(hello|hi|hey|greetings)\b',
            r'\b(good\s(morning|afternoon|evening))\b',
            r'\bhow\s(are\syou|are\sthings|is\sit\sgoing)\b',
            r'\bwhat\'s\sup\b',
            r'\bnice\sto\smeet\syou\b'
        ]
        
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in greeting_patterns)

    def _prepare_retrieved_docs(self, docs: List[Document]) -> str:
        return "\n".join([doc.page_content for doc in docs])


    def _load_chat_history(self, chat_id: str, user_email: str) -> ConversationSummaryMemory:
        chat_history: List[Dict[str, str]] = self.chat_manager.get_chat_history(chat_id, user_email)
        
        if len(chat_history) > 0:
            history = ChatMessageHistory()
            for message in chat_history:
                if message["role"] == "user":
                    history.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    history.add_ai_message(message["content"])

            return ConversationSummaryMemory.from_messages(
                llm=self.llm,
                chat_memory=history,
                return_messages=True
            )

    def _save_message(self, chat_id: str, user_email: str, user_message: str, response: str):
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)