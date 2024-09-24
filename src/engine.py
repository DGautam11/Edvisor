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

        self._system_prompt = """You are an AI assistant for Edvisor, a chatbot specializing in Finland Study and Visa Services. 
        Provide helpful and accurate information only about studying in Finland, Finnish education system, student visas. If you don't know the answer, politely inform the user that you are unable to assist with the query.
        If the query is not related to these topics, politely inform the user that you can only assist with Finland study and visa related queries.
        For simple greetings, respond politely and briefly."""

        self.greeting_prompt = PromptTemplate.from_template(
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self._system_prompt}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {{user_query}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )

        self.full_prompt = PromptTemplate.from_template(
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self._system_prompt}<|eot_id|>
            <|start_header_id|>Conversation Summary:<|end_header_id|> {{context}}Use above information to maintain the context<|eot_id|>
            <|start_header_id|>Retrieved Information:<|end_header_id|> 
            {{retrieved_docs}}
            Use the above information to inform your response.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {{user_query}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        print(f"Generating response for user: {user_email}, message: {user_message}")
        
        # Check if it's a new chat
        is_new_chat = len(self.chat_manager.get_chat_history(chat_id, user_email)) == 0
        print(f"Is new chat: {is_new_chat}")

        # Check if the message is a greeting
        is_greeting = self._is_greeting(user_message)
        print(f"Is greeting: {is_greeting}")

        if is_new_chat or is_greeting:
            print("Using greeting prompt")
            chain = self.greeting_prompt | self.llm
            full_response = chain.invoke({"user_query": user_message})
        else:
            print("Using full prompt with RAG")
            memory = self._load_chat_history(chat_id, user_email)
            retrieved_docs = self.rag.query_vector_store(user_message, k=2)
            retrieved_docs_text = self._prepare_retrieved_docs(retrieved_docs)
            print(f"Retrieved docs: {retrieved_docs_text}")
            
            if not retrieved_docs_text.strip():
                print("No relevant documents found, using fallback")
                retrieved_docs_text = "No specific information found. Using general knowledge about studying in Finland."

            max_tokens = self.config.max_context_length
            system_prompt_tokens = len(self.tokenizer.encode(self._system_prompt))
            user_message_tokens = len(self.tokenizer.encode(user_message))
            retrieved_docs_tokens = len(self.tokenizer.encode(retrieved_docs_text))
            available_context_tokens = max_tokens - system_prompt_tokens - user_message_tokens - retrieved_docs_tokens - 100

            truncated_context = self._truncate_context(memory.buffer, available_context_tokens)
            print(f"Truncated context: {truncated_context}")

            chain = self.full_prompt | self.llm
            full_response = chain.invoke({
                "context": truncated_context,
                "retrieved_docs": retrieved_docs_text,
                "user_query": user_message
            })

        print(f"Full response from model: {full_response}")

        # Extract only the assistant's response
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
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

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        encoded_context = self.tokenizer.encode(context)
        if len(encoded_context) <= max_tokens:
            return context
        
        truncated_encoded = encoded_context[-max_tokens:]
        return self.tokenizer.decode(truncated_encoded)

    def _load_chat_history(self, chat_id: str, user_email: str) -> ConversationSummaryMemory:
        chat_history: List[Dict[str, str]] = self.chat_manager.get_chat_history(chat_id, user_email)
        
        if chat_history:
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