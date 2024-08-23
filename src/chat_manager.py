import uuid
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import Config
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

class ChatManager:
    """
    Manages chat sessions, including conversation history and context retrieval.

    This class handles the creation and management of chat sessions, storing chat
    history in memory and in a vector database (Chroma) for efficient context retrieval.

    Attributes:
        config (Config): Configuration object containing settings.
        chroma_client (chromadb.Client): Client for interacting with the Chroma vector database.
        collection (chromadb.Collection): Chroma collection for storing chat messages.
        active_chats (Dict[str, List[Dict[str, str]]]): In-memory storage of active chat sessions.
    """

    def __init__(self):
        """
        Initialize the ChatManager with configuration and set up the Chroma database.
        """
        self.config = Config()
        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=self.config.chroma_persist_directory
        ))
        
        # Create an embedding function using Chroma's utility
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        # Get or create a collection for storing chat history
        self.collection = self.chroma_client.get_or_create_collection( name = "chat_history",embedding_function = self.embedding_function)
        # Dictionary to store active chat sessions in memory
        self.active_chats: Dict[str, List[Dict[str, str]]] = {}
       
        

    def create_new_chat(self) -> str:
        """
        Create a new chat session.

        Returns:
            str: A unique identifier (UUID) for the new chat session.
        """
        chat_id = str(uuid.uuid4())
        self.active_chats[chat_id] = {
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str):
        """
        Add a message to a specified chat session.

        This method adds the message to both the in-memory storage and the Chroma database.

        Args:
            chat_id (str): The identifier of the chat session.
            role (str): The role of the message sender (e.g., 'user' or 'assistant').
            content (str): The content of the message.

        Raises:
            ValueError: If the specified chat_id does not exist.
        """
        if chat_id not in self.active_chats:
            raise ValueError(f"Chat {chat_id} does not exist.")
        
        message = {"role": role, "content": content}
        self.active_chats[chat_id].append(message)
        self.active_chats[chat_id]["updated_at"] = datetime.now().isoformat()
        
        # Limit chat history to prevent excessive memory usage
        if len(self.active_chats[chat_id]) > self.config.chat_history_limit:
            self.active_chats[chat_id] = self.active_chats[chat_id][-self.config.chat_history_limit:]
        
        # Add message to the Chroma vector database for context retrieval
        self.collection.add(
            documents=[content],
            metadatas=[{"chat_id": chat_id, "role": role}],
            ids=[f"{chat_id}_{len(self.active_chats[chat_id])}"]
        )

    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the chat history for a specified chat session.

        Args:
            chat_id (str): The identifier of the chat session.

        Returns:
            List[Dict[str, str]]: A list of message dictionaries for the chat session.

        Raises:
            ValueError: If the specified chat_id does not exist.
        """
        if chat_id not in self.active_chats:
            raise ValueError(f"Chat {chat_id} does not exist.")
        return self.active_chats[chat_id]

    def get_relevant_context(self, chat_id: str, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant context from the vector database based on the query.

        This method uses the Chroma database to find messages similar to the query
        within the specified chat session.

        Args:
            chat_id (str): The identifier of the chat session.
            query (str): The query to find relevant context for.
            k (int, optional): The number of relevant messages to retrieve. Defaults to 5.

        Returns:
            List[str]: A list of relevant message contents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"chat_id": chat_id}
        )
        return results['documents'][0]

    def format_context(self, chat_id: str, query: str) -> str:
        """
        Format the context for the model input.

        This method combines recent chat history with relevant context retrieved
        from the vector database.

        Args:
            chat_id (str): The identifier of the chat session.
            query (str): The current query to find relevant context for.

        Returns:
            str: A formatted string containing recent history and relevant context.
        """
        # Get the last 5 messages from recent history
        recent_history = self.get_chat_history(chat_id)[-5:]
        # Get relevant context based on the query
        relevant_context = self.get_relevant_context(chat_id, query)
        
        formatted_context = "Previous conversation:\n"
        for message in recent_history:
            formatted_context += f"{message['role'].capitalize()}: {message['content']}\n"
        
        formatted_context += "\nRelevant context:\n"
        for context in relevant_context:
            formatted_context += f"{context}\n"
        
        return formatted_context

    def del_conversation(self, chat_id: str):
        """
        Clear the chat history for a specified chat session.

        This method removes the chat history from both the in-memory storage
        and the Chroma database.

        Args:
            chat_id (str): The identifier of the chat session to clear.
        """
        if chat_id in self.active_chats:
            del self.active_chats[chat_id]
        else:
            raise ValueError(f"Chat {chat_id} does not exist.")
        # Remove all messages for this chat from the Chroma database
        self.collection.delete(where={"chat_id": chat_id})
    def get_all_conversations(self) -> List[Dict]:
        """
        Get a list of all conversations, sorted by creation date (most recent first).

        Returns:
            List[Dict]: A list of dictionaries containing details for all conversations.
        """
        sorted_chats = sorted(self.active_chats.items(), key=lambda x: x[1]["created_at"], reverse=True)
        all_chats = []
        for chat_id, chat_data in sorted_chats:
            messages = chat_data["messages"]
            if messages:
                # Use the first user message as the title, if available
                title = next((msg['content'] for msg in messages if msg['role'] == 'user'), "New Chat")
                preview = messages[-1]["content"][:50] + "..."
            else:
                title = "New Chat"
                preview = "Empty chat"
            
            all_chats.append({
                "id": chat_id,
                "title": title[:30] + "..." if len(title) > 30 else title,  # Truncate long titles
                "preview": preview,
                "created_at": chat_data["created_at"]
            })
        return all_chats