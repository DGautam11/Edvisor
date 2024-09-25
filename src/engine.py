from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from model import Model
from chat_manager import ChatManager
from rag import RAG
from transformers import pipeline
from typing import List, Dict
from langchain.docstore.document import Document
import chromadb
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
        self.tokenizer = tokenizer
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

        self._system_prompt = """
                You are an AI assistant for Edvisor, a chatbot specializing in Finland Study and Visa Services. 
                Provide accurate, helpful, and up-to-date information on studying in Finland, the Finnish education system, student visas, and living in Finland as a student. 
                If a conversation summary is provided, use it to maintain context from the user's previous questions and answers, to avoid redundancy, and to provide more personalized responses.
                If retrieved information is available, incorporate it into your response to ensure the latest data or facts are used. 
                Always prioritize accuracy and clarity in your answers, especially when providing information about visa policies, deadlines, or documentation.
                If unsure or the information is unavailable, direct users to trusted sources such as the Finnish Immigration Service or university websites.
                Respond concisely and politely to simple greetings.
                For off-topic queries, politely inform the user that you specialize in Finland study and visa services.
                """

        self.full_prompt = PromptTemplate.from_template(
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self._system_prompt}<|eot_id|>
            <|start_header_id|>Use the following conversation summary to maintain context in your responses (if available):<|end_header_id|> 
            {{context}}<|eot_id|>
            <|start_header_id|>Use the following retrieved information to provide accurate and up-to-date responses (if available):<|end_header_id|> 
            {{retrieved_docs}}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {{user_query}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
            
            # For ongoing conversations, use the full context and RAG
            memory = self._load_chat_history(chat_id, user_email)
            retrieved_docs = self.rag.query_vector_store(user_message, k=2)
            retrieved_docs_text = self._prepare_retrieved_docs(retrieved_docs)
            context = memory.buffer
            print(context)
            print(retrieved_docs_text)
            

            chain = self.full_prompt | self.llm

            full_response = chain.invoke({
                "context": context,
                "retrieved_docs": retrieved_docs_text,
                "user_query": user_message
            })

            # Extract only the assistant's response
            assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()

            # Save the new message to chat manager
            self._save_message(chat_id, user_email, user_message, assistant_response)

            return assistant_response


    def _prepare_retrieved_docs(self, docs: List[Document]) -> str:
        return " ".join([doc.page_content for doc in docs])


    def _load_chat_history(self, chat_id: str, user_email: str) -> ConversationSummaryMemory:
        chat_history: List[Dict[str, str]] = self.chat_manager.get_chat_history(chat_id, user_email)
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