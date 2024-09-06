from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from transformers import AutoTokenizer, pipeline
from model import Model
from chat_manager import ChatManager
from config import Config
from auth import OAuth

class Engine:
    """
    Engine class for managing text generation and response handling in the Edvisor chatbot using LangChain.

    This class initializes the language model, sets up the LangChain components,
    and provides methods for generating responses using a conversation chain.

    Attributes:
        model (Model): An instance of the Model class for language model management.
        tokenizer (AutoTokenizer): The tokenizer for the language model.
        llm (HuggingFacePipeline): The LangChain wrapper for the Hugging Face pipeline.
        memory (ConversationBufferMemory): LangChain memory component for storing conversation history.
        prompt (ChatPromptTemplate): LangChain prompt template for structuring the conversation.
        conversation (ConversationChain): LangChain conversation chain for managing the dialogue.
        chat_manager (ChatManager): Manager for persistent chat history.
        config (Config): Configuration settings.
        oauth (OAuth): OAuth handler.

    Methods:
        setup_langchain(): Initialize and configure LangChain components.
        generate_response(chat_id, user_message, user_email): Generate a response based on the conversation history.
    """

    def __init__(self):
        """Initialize the Engine with a Model instance and set up LangChain components."""
        self.model = Model()
        self.tokenizer = self.model.set_tokenizer()
        self.chat_manager = ChatManager()
        self.config = Config()
        self.oauth = OAuth()
        self.setup_langchain()

    def setup_langchain(self):
        """
        Set up LangChain components including the language model, memory, prompt, and conversation chain.
        """
        # Set up the Hugging Face pipeline
        model, _ = self.model.get_model_tokenizer()
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Create LangChain HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Set up memory
        self.memory = ConversationBufferMemory(return_messages=True)

        # Set up prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant for Edvisor, a chatbot for Finland Study and Visa Services. Provide helpful and accurate information."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        # Set up conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt
        )

    def generate_response(self, chat_id: str, user_message: str, user_email: str):
        """
        Generate a response based on the conversation history using LangChain ConversationChain.

        Args:
            chat_id (str): The ID of the current chat.
            user_message (str): The latest message from the user.
            user_email (str): The email of the current user.

        Returns:
            str: Generated response text.
        """
        # Retrieve chat history from persistent storage
        chat_history = self.chat_manager.get_chat_history(chat_id, user_email)
        chat_history_limit = int(self.config.chat_history_limit)
        recent_messages = chat_history[-chat_history_limit:]

        # Clear the conversation memory and add recent messages
        self.memory.clear()
        for message in recent_messages:
            if message["role"] == "user":
                self.memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                self.memory.chat_memory.add_ai_message(message["content"])

        # Generate response using the ConversationChain
        response = self.conversation.predict(input=user_message)

        # Save the new messages to persistent storage
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)

        return response