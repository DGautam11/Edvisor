import uuid
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import Config
from collections import defaultdict

@dataclass
class ChatData:
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int) -> List[Dict[str, str]]:
        return self.messages[-limit:]

    @property
    def title(self) -> str:
        if self.messages:
            first_user_message = next((msg['content'] for msg in self.messages if msg['role'] == 'user'), "New Chat")
            return first_user_message[:30] + "..." if len(first_user_message) > 30 else first_user_message
        return "New Chat"

class ChatManager:
    def __init__(self,chroma_client):
        self.config = Config()
        self.chroma_client = chroma_client
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        self.user_collection = {}
        self.active_chats: Dict[str,Dict[str, ChatData] ] = {}

    def get_user_collection(self, user_email: str):
        if user_email not in self.user_collection:
            collection_name = f"chat_history_{user_email.replace('@', '_at_')}"
            self.user_collection[user_email] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        return self.user_collection[user_email]

    def create_new_chat(self) -> str:
        chat_id = str(uuid.uuid4())
        self.active_chats[chat_id] = ChatData()
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str, user_email:str):
        collection = self.get_user_collection(user_email)
        message_id = str(uuid.uuid4())
        metadata = {
            "chat_id": chat_id,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[f"{chat_id}_{message_id}"]
            )
        
        self.active_chats[chat_id].add_message(role, content)
        
        
    def get_chat_history(self, chat_id: str,user_email:str) -> List[Dict[str, str]]:
        if chat_id  not in self.active_chats:
            
            # Fetch chat history from Chroma if not in active chats
            collection = self.get_user_collection(user_email)
            results = collection.get(
                where={"chat_id": chat_id},
                include=['metadatas', 'documents']
                )
            messages = [
                {"role": meta['role'], "content": doc,"created_at": meta['created_at']}
                for meta, doc in zip(results['metadatas'], results['documents'])
                ]
            sorted_messages = sorted(messages, key=lambda x: x['created_at'])

            self.active_chats[chat_id] = ChatData(messages=sorted_messages)

        return self.active_chats[chat_id].messages


    def del_conversation(self, chat_id: str,user_email:str):
        collection = self.get_user_collection(user_email)
        collection.delete(where={"chat_id": chat_id})
        if chat_id in self.active_chats:
            del self.active_chats[chat_id]


    def get_all_conversations(self, user_email: str) -> List[Dict]:
        collection = self.get_user_collection(user_email)
        results = collection.get(
            where={},  # Get all documents for this user
            include=['metadatas', 'documents']
        )
        
        # Group messages by chat_id
        chat_messages = defaultdict(list)
        for meta, doc in zip(results['metadatas'], results['documents']):
            chat_messages[meta['chat_id']].append({
                "role": meta['role'],
                "content": doc,
                "created_at": meta['created_at']
            })
        
        conversations = []
        for chat_id, messages in chat_messages.items():
            # Sort messages by timestamp
            sorted_messages = sorted(messages, key=lambda x: x['created_at'])
            
            # Create ChatData object
            chat_data = ChatData(messages=[
                {"role": msg['role'], "content": msg['content']}
                for msg in sorted_messages
            ])
            
            # Get the first message's timestamp as the creation time
            created_at = sorted_messages[0]['created_at'] if sorted_messages else None
            
            conversations.append({
                "id": chat_id,
                "title": chat_data.title,
                "created_at": created_at
            })
        
        # Sort conversations by creation time, most recent first
        return sorted(conversations, key=lambda x: x['created_at'], reverse=True)