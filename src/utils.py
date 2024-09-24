from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone
import chromadb
from chromadb.config import Settings
from rag import RAG
from config import Config
import os
import shutil

class Utils:
    @staticmethod
    def get_relative_time(date_str):
        now = datetime.now(timezone.utc)
        then = parser.parse(date_str)
        delta = relativedelta(now, then)

        if delta.days == 0:
            relative_time = "today"
        elif delta.days == 1:
            relative_time = "yesterday"
        elif delta.days < 7:
            relative_time = f"{delta.days} days ago"
        elif delta.weeks == 1:
            relative_time = "1 week ago"
        elif delta.weeks < 4:
            relative_time = f"{delta.weeks} weeks ago"
        elif delta.months == 1:
            relative_time = "1 month ago"
        elif delta.months < 12:
            relative_time = f"{delta.months} months ago"
        else:
            relative_time = f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
        
        return relative_time
    
    @staticmethod
    def build_rag_database():
        
        config = Config()
        
        
        chroma_client = chromadb.PersistentClient(path=config.chroma_persist_directory)
        try:
            # Initialize RAG
            rag = RAG(chroma_client)
            
            # Build the RAG store
            rag.build_rag_store(config.rag_dataset_path)
            
            print(f"RAG database built successfully.")
            print(f"Vector store saved to {config.chroma_persist_directory}")

            # Optionally, you can add some test queries here
            test_queries = [
                "Why Finland?",
                "Programs at LAB University of Applied Sciences",
                "Tution fee at Turku University of Applied Sciences",
            ]

            print("\nTesting RAG database with sample queries:")
            for query in test_queries:
                print(f"\nQuery: {query}")
                results = rag.search(query, k=3)
                for i, doc in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"Context: {doc.metadata['context']}")
                    print(f"Source: {doc.metadata['source']}")
                    print(f"Similarity Score: {doc.metadata['similarity_score']:.4f}")
                    print(f"Content: {doc.page_content[:200]}...")
                    print("---")

        finally:
            print("Chroma client closed.")
        
    @staticmethod
    def initialize_chroma_db():
        config = Config()
        chroma_client = chromadb.PersistentClient(path=config.chroma_persist_directory)
        
        # Try to create a test collection to ensure the database is initialized
        try:
            test_collection = chroma_client.create_collection("test_collection")
            test_collection.add(
                documents=["This is a test document"],
                metadatas=[{"source": "test"}],
                ids=["test1"]
            )
            print("ChromaDB initialized successfully.")
            
            # Clean up the test collection
            chroma_client.delete_collection("test_collection")
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")