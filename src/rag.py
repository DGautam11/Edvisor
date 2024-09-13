import json
import os
import re
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
        print(f"RAG initialized with embedding model: {self.config.embedding_model}")

    def load_json_data(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Successfully loaded JSON data from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading JSON data from {file_path}: {str(e)}")
            return {}

    def dict_to_string(self, data: Any, indent: str = "") -> str:
        if isinstance(data, dict):
            return "\n".join(f"{indent}{k}: {self.dict_to_string(v, indent + '  ')}" for k, v in data.items())
        elif isinstance(data, list):
            return "\n".join(f"{indent}- {self.dict_to_string(item, indent + '  ')}" for item in data)
        else:
            return str(data)

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def process_json_data(self, data: Dict, file_name: str) -> List[Document]:
        documents = []
        university_name = data.get('university', '')
        short_name = data.get('short name', '')
        print(f"Processing JSON data for {university_name} ({short_name})")

        # Process university info
        university_info = {
            k: v for k, v in data.items()
            if k in ['university', 'short name', 'about'] or k.startswith('contact')
        }
        if university_info:
            context = f"University Information: {university_name} ({short_name})"
            content = self.dict_to_string(university_info)
            documents.append(Document(
                page_content=f"{context}\n\n{content}",
                metadata={"context": context, "source": file_name}
            ))
            print(f"Created document for university information")

        # Process degree programs
        for degree_type in ['bachelor\'s programs', 'master\'s programs']:
            if degree_type in data:
                for program in data[degree_type]:
                    program_name = program.get('program')
                    credits = program.get('credits')
                    duration = program.get('duration')
                    language = program.get('language')
                    
                    context = (f"{degree_type.capitalize()} at {university_name} ({short_name}): "
                               f"{program_name}")

                    additional_info = []
                    if credits:
                        additional_info.append(f"{credits} credits")
                    if duration:
                        additional_info.append(f"{duration}")
                    if language:
                        additional_info.append(f"Language: {language}")

                    if additional_info:
                        context += " - " + ", ".join(additional_info)

                    print(f"Creating document for program: {context}")

                    content = self.dict_to_string(program)
                    documents.append(Document(
                        page_content=f"{context}\n\n{content}",
                        metadata={"context": context, "source": file_name}
                    ))

        # Process other sections
        for key, value in data.items():
            if key not in ['university', 'short name', 'about', 'bachelor\'s programs', 'master\'s programs'] and not key.startswith('contact'):
                context = f"{key.capitalize()} at {university_name} ({short_name})"
                content = self.dict_to_string(value)
                documents.append(Document(
                    page_content=f"{context}\n\n{content}",
                    metadata={"context": context, "source": file_name}
                ))
                print(f"Created document for section: {key}")

        print(f"Total documents created for {university_name}: {len(documents)}")
        return documents

    def process_text_file(self, file_path: str) -> List[Document]:
        print(f"Processing text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        documents = []
        sections = content.split("Context:")

        for i, section in enumerate(sections[1:], 1):
            lines = section.strip().split('\n')
            context = lines[0].strip()
            text = '\n'.join(lines[1:])

            chunks = self.text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=f"{context}\n\n{chunk}",
                    metadata={
                        "context": context,
                        "source": os.path.basename(file_path)
                    }
                ))
            print(f"Created {len(chunks)} chunks for context: {context}")

        print(f"Total documents created from {file_path}: {len(documents)}")
        return documents

    def create_vector_store(self, documents: List[Document]):
        print(f"Creating vector store with {len(documents)} documents")
        for i, doc in enumerate(documents):
            self.rag_collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"]
            )
        print("Vector store creation completed")

    def build_rag_store(self, directory_path: str):
        print(f"Building RAG store from directory: {directory_path}")

        existing_collections = self.chroma_client.list_collections()
        if any(collection.name == "rag" for collection in existing_collections):
            self.chroma_client.delete_collection("rag")
            print("Deleted existing 'rag' collection.")

        self.rag_collection = self.chroma_client.create_collection(
            name="rag",
            embedding_function=self.embedding_function
        )
        print("Created new 'rag' collection.")

        all_documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith('.json'):
                print(f"Processing JSON file: {filename}")
                json_data = self.load_json_data(file_path)
                processed_docs = self.process_json_data(json_data, filename)
                all_documents.extend(processed_docs)
                print(f"Processed {filename}: {len(processed_docs)} documents created")
            elif filename.endswith('.txt'):
                print(f"Processing text file: {filename}")
                processed_docs = self.process_text_file(file_path)
                all_documents.extend(processed_docs)
                print(f"Processed {filename}: {len(processed_docs)} documents created")

        self.create_vector_store(all_documents)
        
        print(f"RAG vector store created with {len(all_documents)} documents.")
        print(f"Vector store saved to {self.config.rag_persist_chroma_directory}")

    def query_vector_store(self, query: str, k: int = 5):
        print(f"Querying vector store with: '{query}'")
        processed_query = self.preprocess_text(query)
        results = self.rag_collection.query(
            query_texts=[processed_query],
            n_results=k
        )
        
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
        print(f"Retrieved {len(documents)} documents from vector store")
        return documents

    def inspect_vector_store(self):
        total_docs = self.rag_collection.count()
        print(f"Total documents in the vector store: {total_docs}")
        
        sample = self.rag_collection.get(limit=10, include=["metadatas"])
        
        print("\nSample of stored documents:")
        for id, metadata in zip(sample['ids'], sample['metadatas']):
            print(f"ID: {id}")
            print(f"Metadata: {metadata}")
            print("---")

    def search(self, query: str, k: int = 5):
        results = self.query_vector_store(query, k)
        print(f"\nSearch results for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Context: {doc.metadata['context']}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content[:200]}...")
            print("---")