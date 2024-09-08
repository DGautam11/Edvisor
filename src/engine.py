from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, RunnablePassThrough
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
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
        Use the conversation summary to inform your responses and maintain context. If you don't know the answer, politely inform the user that you are unable to assist with the query.
        If the query is not related to these topics, politely inform the user that you can only assist with Finland study and visa related queries.
        For simple greetings, respond politely and briefly."""

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self._system_prompt}<|eot_id|>
                        <|start_header_id|>Conversation Summary:<|end_header_id|> {{context}}><|eot_id|>
                      <|start_header_id|>user<|end_header_id|>
                      {{user_query}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """

        self.prompt = PromptTemplate.from_template(template)

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        # Load chat history and create memory
        memory = self._load_chat_history(chat_id, user_email)
        
        # Create chain with the loaded memory
        chain = ({"context":memory.buffer,"user_query":RunnablePassThrough()}) | self.prompt | self.llm
        
        
        # Generate response
        response = chain.invoke(user_message)
        
        # Save the new interaction to memory
        memory.save_context({"input": user_message}, {"output": response})
        
        # Save the new message to chat manager
        self._save_message(chat_id, user_email, user_message, response)

        print(response)

        return "Hello"
    

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

