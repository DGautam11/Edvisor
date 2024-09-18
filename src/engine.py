from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.docstore.document import Document
from model import Model
from chat_manager import ChatManager
from transformers import pipeline
from typing import List, Dict

class Engine:
    def __init__(self):
        self.model = Model()
        self.chat_manager = ChatManager()
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
        Provide helpful and accurate information only about studying in Finland, Finnish education system, student visas, and living in Finland as a student. 
        Use the conversation summary and retrieved information to inform your responses and maintain context. If you don't know the answer, politely inform the user that you are unable to assist with the query.
        If the query is not related to these topics, politely inform the user that you can only assist with Finland study and visa related queries.
        For simple greetings, respond politely and briefly."""

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self._system_prompt}<|eot_id|>
                        <|start_header_id|>Conversation Summary:<|end_header_id|> {{context}}<|eot_id|>
                        <|start_header_id|>Retrieved Information:<|end_header_id|> {{retrieved_docs}}<|eot_id|>
                        <|start_header_id|>user<|end_header_id|>
                        {{user_query}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """

        self.prompt = PromptTemplate.from_template(template)

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        # Load chat history and create memory
        memory = self._load_chat_history(chat_id, user_email)
         # Retrieve the conversation history
        memory_variables = memory.load_memory_variables({})
        conversation_history = memory_variables.get('history', '')
        print(conversation_history)

        # Retrieve relevant documents
        retrieved_docs = self.rag.query_vector_store(user_message, k=1)

        # Prepare retrieved documents text
        retrieved_docs_text = self._prepare_retrieved_docs(retrieved_docs)
        # Calculate available tokens for context
        max_tokens = 4096  # Context length for Meta-Llama-3-8B-Instruct
        system_prompt_tokens = len(self.tokenizer.encode(self._system_prompt))
        user_message_tokens = len(self.tokenizer.encode(user_message))
        retrieved_docs_tokens = len(self.tokenizer.encode(retrieved_docs_text))
        available_context_tokens = max_tokens - system_prompt_tokens - user_message_tokens - retrieved_docs_tokens - 100  # Buffer

        # Truncate conversation history if needed
        truncated_context = self._truncate_context(memory.buffer, available_context_tokens)

        # Create chain with the loaded memory
        chain =  self.prompt | self.llm
        
       # Generate response
        full_response = chain.invoke({
            "context": truncated_context,
            "retrieved_docs": retrieved_docs_text,
            "user_query": user_message
        })
    
        # Extract only the assistant's response
        assistant_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    
        # Save the new interaction to memory
        memory.save_context({"input": user_message}, {"output": assistant_response})
    
        # Save the new message to chat manager
        self._save_message(chat_id, user_email, user_message, assistant_response)

        print(assistant_response)

        return assistant_response
    
    def _prepare_retrieved_docs(self, docs: List[Document]) -> str:
        return " ".join([doc.page_content for doc in docs])

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        encoded_context = self.tokenizer.encode(context)
        if len(encoded_context) <= max_tokens:
            return context
        
        truncated_encoded = encoded_context[-max_tokens:]
        return self.tokenizer.decode(truncated_encoded)
    
    def _load_chat_history(self, chat_id: str, user_email: str) -> ConversationSummaryMemory:
        chat_history: List[Dict[str, str]] = self.chat_manager.get_chat_history(chat_id, user_email)
        
        if not chat_history:
            # If chat history is empty, create a new ConversationSummaryMemory
            return ConversationSummaryMemory(llm=self.llm, max_token_limit=256)
        else:
            # If there's chat history, use from_messages to create the memory
            history = ChatMessageHistory()
            for message in chat_history:
                if message["role"] == "user":
                    history.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    history.add_ai_message(message["content"])

            return ConversationSummaryMemory.from_messages(
                llm=self.llm,
                chat_memory=history,
                max_token_limit=256

            )

    def _save_message(self, chat_id: str, user_email: str, user_message: str, response: str):
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)

