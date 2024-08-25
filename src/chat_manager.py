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

    def get_recent_messages(self, limit: int = 5) -> List[Dict[str, str]]:
        return self.messages[-limit:]

    @property
    def title(self) -> str:
        if self.messages:
            first_user_message = next((msg['content'] for msg in self.messages if msg['role'] == 'user'), "New Chat")
            return first_user_message[:30] + "..." if len(first_user_message) > 30 else first_user_message
        return "New Chat"

class ChatManager:
    def __init__(self):
        self.config = Config()
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=self.config.chroma_persist_directory
        ))
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="chat_history",
            embedding_function=self.embedding_function
        )
        self.active_chats: Dict[str, ChatData] = {}

    def create_new_chat(self) -> str:
        chat_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        self.active_chats[chat_id] = ChatData(created_at=created_at)
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str):
        if chat_id not in self.active_chats:
            self.create_new_chat()
        
        self.active_chats[chat_id].add_message(role, content)
        
        self.collection.add(
            documents=[content],
            metadatas=[{
                "chat_id": chat_id,
                "role": role,
            }],
            ids=[f"{chat_id}_{len(self.active_chats[chat_id].messages)}"]
        )

    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        if chat_id not in self.active_chats:
            # Fetch chat history from Chroma if not in active chats
            results = self.collection.get(
                where={"chat_id": chat_id},
                include=['metadatas', 'documents']
            )
            messages = [
                {"role": meta['role'], "content": doc}
                for meta, doc in zip(results['metadatas'], results['documents'])
            ]
            self.active_chats[chat_id] = ChatData(messages=messages)
        return self.active_chats[chat_id].messages

    def get_relevant_context(self, chat_id: str, query: str, k: int = 5) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"chat_id": chat_id}
        )
        return results['documents'][0] if results['documents'] else []

    def format_context(self, chat_id: str, query: str) -> str:
        if chat_id not in self.active_chats:
            self.get_chat_history(chat_id)
        
        recent_history = self.active_chats[chat_id].get_recent_messages()
        relevant_context = self.get_relevant_context(chat_id, query)
        
        formatted_context = "Previous conversation:\n"
        for message in recent_history:
            formatted_context += f"{message['role'].capitalize()}: {message['content']}\n"
        
        formatted_context += "\nRelevant context:\n"
        for context in relevant_context:
            formatted_context += f"{context}\n"
        
        return formatted_context

    def del_conversation(self, chat_id: str):
        if chat_id in self.active_chats:
            del self.active_chats[chat_id]
        self.collection.delete(where={"chat_id": chat_id})

    def get_all_conversations(self) -> List[Dict]:
        all_messages = self.collection.get(
            include=['metadatas', 'documents']
        )

        chats = defaultdict(list)
        for metadata, document in zip(all_messages['metadatas'], all_messages['documents']):
            chat_id = metadata['chat_id']
            chats[chat_id].append({
                'role': metadata['role'],
                'content': document
            })

        all_conversations = []
        for chat_id, messages in chats.items():
            title = next((msg['content'] for msg in messages if msg['role'] == 'user'), "New Chat")
            title = title[:30] + "..." if len(title) > 30 else title
            
            # Get the created_at timestamp from active_chats if available, otherwise use current time
            created_at = self.active_chats[chat_id].created_at if chat_id in self.active_chats else datetime.now(timezone.utc).isoformat()

            all_conversations.append({
                "id": chat_id,
                "title": title,
                "created_at": created_at
            })

        return sorted(all_conversations, key=lambda x: x['created_at'], reverse=True)