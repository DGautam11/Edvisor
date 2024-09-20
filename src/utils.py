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

    def __init__(self):
        self.config = Config()
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
    def build_rag_database(self):
        
        # Create a Chroma client
        chroma_client = chromadb.Client(Settings(
            persist_directory=self.config.chroma_persist_directory,
        ))
        
        try:
            # Initialize RAG
            rag = RAG(chroma_client)
            
            # Build the RAG store
            rag.build_rag_store(self.config.rag_dataset_path)
            
            print(f"RAG database built successfully.")
            print(f"Vector store saved to {self.config.chroma_persist_directory}")

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
    
    def backup_chroma_db(self):
        """
        Manually backup the local ChromaDB to Google Drive.
    
        """
        local_path = self.chroma_persist_directory
        backup_path = self.chroma_db_backup_path

        print(f"Backing up ChromaDB from {local_path} to {backup_path}")
        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"No ChromaDB found at {local_path} or directory is empty. Nothing to backup.")
            return False
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(local_path, backup_path)
            print(f"Backup completed. Files copied to: {backup_path}")
        except Exception as e:
            print(f"Error during backup: {str(e)}")
    
    @staticmethod
    def restore_chroma_db():
        """
        Restore the ChromaDB from Google Drive backup to local path.
        """
        local_path = Config().chroma_persist_directory
        backup_path = Config().chroma_db_backup_path

        if not os.path.exists(backup_path) or not os.listdir(backup_path):
            print(f"No backup found at {backup_path} or backup directory is empty.")
            return False

        print(f"Restoring ChromaDB from {backup_path} to {local_path}")
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.copytree(backup_path, local_path)
            print(f"Restore completed. Files copied to: {local_path}")
            return True
        except Exception as e:
            print(f"Error during restore: {str(e)}")
            return False
