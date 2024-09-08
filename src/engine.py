from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
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
        self.llm = HuggingFacePipeline(pipeline =hf_pipeline)

        #Prompt for summarization
        summarize_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Summarize the key points from the conversation history to provide context for the user query. Dont't create your own context. if their is none provide a empty string"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Provide a concise summary of the relevant points discussed so far.")
        ])

        # Chain for summarization
        self.summarize_chain = LLMChain(
            llm=self.llm,
            prompt=summarize_prompt,
            verbose=True,
            output_key="context_summary"
        )

        #Prompt for query generation with previous context
        query_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Based on the conversation summary and the current user query, create a comprehensive query that captures the user's intent and relevant context. 
           """),
            HumanMessagePromptTemplate.from_template("Context summary: {context_summary}\nCurrent query: {input}\n\nGenerate a detailed query that encompasses the user's question and relevant background information.")
        ])

        # Chain for query generation
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=query_prompt,
            verbose=True,
            output_key="detailed_query"
        )

        #Prompt for response generation
        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant for Edvisor, a chatbot specializing in Finland Study and Visa Services. 
            Provide helpful and accurate information only about studying in Finland, Finnish education system, student visas, and living in Finland as a student. 
            If the query is not related to these topics, politely inform the user that you can only assist with Finland study and visa related queries."""),
            HumanMessagePromptTemplate.from_template("{detailed_query}")
        ])

        # Chain for final response
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=response_prompt,
            verbose=True,
            output_key="response"
        )

        # Sequential chain to combine summarization, context extraction, and response generation
        self.conversation_chain = SequentialChain(
            chains=[self.summarize_chain, self.query_chain, self.response_chain],
            input_variables=["chat_history", "input"],
            output_variables=["response"],
            verbose=True
        )

    def _load_chat_history_to_memory(self,chat_id:str,user_email:str):

        chat_history = self.chat_manager.get_chat_history(chat_id,user_email)
        history = ChatMessageHistory()
        for message in chat_history:
            if message["role"] == "user":
                history.add_user_message(message["content"])
            else:
                history.add_ai_message(message["content"])
            
        return ConversationBufferMemory(chat_memory=history,return_messages=True)
        
        
    def generate_response(self, chat_id: str, user_email: str, user_message: str):
        
        memory = self._load_chat_history_to_memory(chat_id, user_email)

        chat_history = memory.chat_memory.messages

            # Generate response using the SequentialChain
        chain_response = self.conversation_chain({
            "chat_history": chat_history,
            "input": user_message
            })

        response = chain_response["response"]

        # Save the new messages to persistent storage and update active_chats
        self.chat_manager.add_message(chat_id, "user", user_message, user_email)
        self.chat_manager.add_message(chat_id, "assistant", response, user_email)

        return response



