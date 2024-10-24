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
import re

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
        self.chat_memories = {}
    
    def _setup_llm(self):
        model, tokenizer = self.model.get_model_tokenizer()
        self.tokenizer = tokenizer
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
    
    def _system_prompt(self):
        return """
            You are an AI assistant named Edvisor, a chatbot specializing in Finland Study and Visa Services. 
            Provide accurate, helpful, and up-to-date information on studying in Finland, the Finnish education system, student visas, and living in Finland as a student. 
            Always respond to user queries, even if they are complex or require detailed information.
            If a conversation summary is provided, use it to maintain context from the user's previous questions and answers, to avoid redundancy, and to provide more personalized responses.
            If retrieved information is available, incorporate it into your response to ensure the latest data or facts are used. 
            Always prioritize accuracy and clarity in your answers, especially when providing information about visa policies, deadlines, or documentation.
            If unsure or the information is unavailable, say so and suggest where the user might find more information.
            Respond concisely to simple greetings, but provide detailed answers to complex questions about studying in Finland.
            For off-topic queries, politely inform the user that you specialize in Finland study and visa services, but still attempt to provide a helpful response.
            """
        
    def _is_greeting(self, message: str) -> bool:
  
        greeting_patterns = [
            r'\b(hello|hi|hey|greetings)\b',         # Simple greetings
            r'\b(good\s(morning|afternoon|evening))\b',  # Time-specific greetings
            r'\bhow\s(are\syou|is\sit\sgoing)\b',    # Common follow-up questions
            r'\bwhat\'s\sup\b',
            r'\bnice\sto\smeet\syou\b'
        ]
    
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in greeting_patterns)
    
    def _greeting_chain(self):
        greeting_prompt = PromptTemplate.from_template(
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant named Edvisor, a chatbot specializing in Finland Study and Visa Services. 
        Provide accurate, helpful, and up-to-date information on studying in Finland, the Finnish education system, student visas, and living in Finland as a student.. The user has just greeted you. 
        Respond with a friendly and polite greeting message, offering your assistance in a helpful manner.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    )
        chain = greeting_prompt | self.llm
        return chain
    
    def _handle_greeting(self, user_message: str)-> str:
        chain = self._greeting_chain()
        response = chain.invoke({
            "user_query": user_message
        })
        assistant_response = self._extract_assistant_response(response)
        return assistant_response
       
        
    

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
            
            print(f"User message: {user_message}")

            memory = self._get_or_create_memory(chat_id,user_email)
            
            # Check if the message is a greeting
            if self._is_greeting(user_message):
                assistant_response = self._handle_greeting(user_message)
                self._save_message(chat_id, user_email, user_message, assistant_response)
                # Update the memory with the new interaction
                memory.save_context({"input": user_message}, {"output": assistant_response})
                print(f"assistant response: {assistant_response}")
                return assistant_response
            
            # Load chat history and update memory
            prev_conversation_summary = memory.buffer

            retrieved_docs = self.rag.query_vector_store(user_message, k=1)
            retrieved_docs_content = self._prepare_retrieved_docs(retrieved_docs)

            print(f"Previous conversation summary: {prev_conversation_summary}")
            print(f"Retrieved documents: {retrieved_docs_content}")

            
            prompt = PromptTemplate.from_template (
                """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>
                <|start_header_id|>Use the following previous conversation summary to maintain context in your responses 
                (if available):<|end_header_id|> 
                {previous_conversation_summary}<|eot_id|>
                <|start_header_id|>Use the following retrieved information to provide accurate and up-to-date responses 
                (if available):<|end_header_id|> 
                {retrieved_docs}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """ )
            
            chain = prompt | self.llm

            full_response = chain.invoke({
                "system_prompt": self._system_prompt(),
                "previous_conversation_summary": prev_conversation_summary,
                "retrieved_docs": retrieved_docs_content,
                "user_query": user_message
            })
            print(f"full response: {full_response}")

            # Extract only the assistant's response
            assistant_response = self._extract_assistant_response(full_response)

            # Save the new message to chat manager
            self._save_message(chat_id, user_email, user_message, assistant_response)
            memory.save_context({"input": user_message}, {"output": assistant_response})

            print(f"assistant response: {assistant_response}")

            return assistant_response
    
    def _get_or_create_memory(self,chat_id:str,user_email:str)->ConversationSummaryMemory:
        if chat_id not in self.chat_memories:
           chat_history = self.chat_manager.get_chat_history(chat_id, user_email)
           history = ChatMessageHistory()
           if chat_history:
               for message in chat_history:
                   if message["role"] == "user":
                       history.add_user_message(message["content"])
                   elif message["role"] == "assistant":
                        history.add_assistant_message(message["content"])
           self .chat_memories[chat_id] = ConversationSummaryMemory.from_messages(
                llm = self.llm,
                chat_memory = history,
                return_messages = True
            )
        return self.chat_memories[chat_id]

    def _prepare_retrieved_docs(self, docs: List[Document]) -> str:
        return "\n".join([doc.page_content for doc in docs])

    def _extract_assistant_response(self, full_response: str) -> str:
        """Extracts the assistant's response from the full LLM output."""
        return full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()


    def _save_message(self, chat_id: str, user_email: str, user_message: str, response: str):
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)