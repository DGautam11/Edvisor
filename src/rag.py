import json
import os
from typing import List, Dict, Any
from config import Config
from langchain.docstore.document import Document
from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAG:
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=self.config.rag_persist_chroma_directory,
        ))
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        self.rag_collection = self.chroma_client.get_or_create_collection(
            name="rag",
            embedding_function=self.embedding_function
        )

    def load_json_data(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def dict_to_string(self, data: Any, indent: str = "") -> str:
        if isinstance(data, dict):
            return "\n".join(f"{indent}{k}: {self.dict_to_string(v, indent + '  ')}" for k, v in data.items())
        elif isinstance(data, list):
            return "\n".join(f"{indent}- {self.dict_to_string(item, indent + '  ')}" for item in data)
        else:
            return str(data)

    def process_json_data(self, data: Dict, file_name: str) -> List[Document]:
        documents = []
        university_name = data['university name']

        # Process university info
        university_info = {
            k: v for k, v in data.items() 
            if k in ['university name', 'short name', 'about'] or k.startswith('contact')
        }
        if university_info:
            documents.append(Document(
                page_content=self.dict_to_string(university_info),
                metadata={"type": "university_info", "university": university_name, "source": file_name}
            ))

        # Process all other key-value pairs
        for key, value in data.items():
            if key not in university_info:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            documents.append(Document(
                                page_content=self.dict_to_string(item),
                                metadata={
                                    "type": key,
                                    "university": university_name,
                                    "source": file_name,
                                    "program": item['program'] 
                                }
                            ))
                        else:
                            documents.append(Document(
                                page_content=str(item),
                                metadata={"type": key, "university": university_name, "source": file_name}
                            ))
                elif isinstance(value, dict):
                    documents.append(Document(
                        page_content=self.dict_to_string(value),
                        metadata={"type": key, "university": university_name, "source": file_name}
                    ))
                else:
                    documents.append(Document(
                        page_content=str(value),
                        metadata={"type": key, "university": university_name, "source": file_name}
                    ))

        return documents

    def create_vector_store(self, documents: List[Document]):
        for i, doc in enumerate(documents):
            self.rag_collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"]
            )

    def process_text_file(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        documents = []
        current_context = ""
        current_text = ""

        for line in content.split('\n'):
            if line.startswith("Context:"):
                if current_text:
                    chunks = self.text_splitter.split_text(current_text)
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(file_path),
                                "context": current_context,
                                "chunk_id": i
                            }
                        ))
                current_context = line.replace("Context:", "").strip()
                current_text = ""
            else:
                current_text += line + "\n"

        # Process the last section
        if current_text:
            chunks = self.text_splitter.split_text(current_text)
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(file_path),
                        "context": current_context,
                        "chunk_id": i
                    }
                ))

        return documents

    def create_vector_store(self, documents: List[Document]):
        for i, doc in enumerate(documents):
            self.rag_collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"]
            )

    def build_rag_store(self, directory_path: str):
        all_documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith('.json'):
                json_data = self.load_json_data(file_path)
                all_documents.extend(self.process_json_data(json_data, filename))
            elif filename.endswith('.txt'):
                all_documents.extend(self.process_text_file(file_path))

        self.create_vector_store(all_documents)
        
        print(f"RAG vector store created with {len(all_documents)} documents.")
        print(f"Vector store saved to {self.config.rag_persist_directory}")
        
    def query_vector_store(self, query: str, k: int = 5):
        results = self.rag_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]