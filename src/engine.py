from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, AIMessage
from model import Model
from chat_manager import ChatManager
from transformers import pipeline

class Engine:
    def __init__(self):
        self.model = Model()
        self.chat_manager = ChatManager()
        self._conversation_pipeline()
    
    def _conversation_pipeline(self):
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

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant for Edvisor, a chatbot specializing in Finland Study and Visa Services. 
            Provide helpful and accurate information only about studying in Finland, Finnish education system, student visas, and living in Finland as a student. 
            If the query is not related to these topics, politely inform the user that you can only assist with Finland study and visa related queries.
            For simple greetings, respond politely and briefly."""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferMemory(return_messages=True)
        )

    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        # Load chat history into memory
        chat_history = self.chat_manager.get_chat_history(chat_id, user_email)
        self.conversation.memory.chat_memory.clear()  # Clear existing memory
        for message in chat_history:
            if message["role"] == "user":
                self.conversation.memory.chat_memory.add_user_message(message["content"])
            else:
                self.conversation.memory.chat_memory.add_ai_message(message["content"])

        
        # Generate response
        response = self.conversation.predict(input=user_message)
        print(response)
        print(type(response))
        
       
        

        # Save the new messages to persistent storage
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)

        return response