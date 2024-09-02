import os
import json
from dataclasses import dataclass, field
from transformers import BitsAndBytesConfig
import torch

@dataclass
class Config:
    """
    Configuration class for the Edvisor project.

    This class manages the configuration settings for the Edvisor chatbot,
    including API keys, model settings, and file paths. It is responsible for
    loading API keys from a JSON file and setting up necessary directory structures.

    Attributes:
        base_path (str): The root directory of the project.
        HF_token (str): The HuggingFace API token.
        google_api_key (str): The Google API key.
        google_cse_id (str): The Google Custom Search Engine ID.
        base_model (str): The name/path of the base language model.
        embedding_model (str): The name/path of the embedding model.
        dataset_path (str): Path to the dataset directory.
        chroma_db_path (str): Path to the Chroma DB directory.
        api_keys_file (str): Name of the file containing API keys.

    Methods:
        get_base_path(): Determines the base path of the project.
        setup_paths(): Sets up necessary directory structures.
        load_api_keys(): Loads API keys from the JSON file.
        validate_api_keys(): Ensures all required API keys are present.
        to_dict(): Converts the configuration to a dictionary.

    Raises:
        FileNotFoundError: If the API keys file is not found.
        ValueError: If required API keys are missing from the file.

    Usage:
        config = Config()
        hf_token = config.HF_token
        dataset_dir = config.dataset_path
    """
    base_path: str = field(default="",init=False)

    #API Keys (initalized later)
    HF_token: str = field(default="",init=False)
    google_api_key: str = field(default="",init=False)
    google_cse_id: str = field(default="",init=False)
    client_id: str = field(default="",init=False)
    client_secret: str = field(default="",init=False)
    auth_uri: str = field(default="",init=False)
    token_uri: str = field(default="",init=False)
    scopes:list = field(default="",init=False)
    redirect_uris: list = field(default="",init=False)

    #Model Configurations
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    #Quantization Configurations
    quant_config: BitsAndBytesConfig = field(default_factory=lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ))

    #Name of the API keys file
    api_keys_file: str = "api_keys.json"

    chroma_persist_directory: str = field(default="",init=False)
    max_context_length: int = 2048
    chat_history_limit: int = 20

    oauth_credentials_file: str = "oauth_credentials.json"


    def __post_init__(self):
        """
        Initialize the configuration after the dataclass is instantiated.
        This method sets up the base path, initializes directory structures,
        loads API keys, and prints the configuration.
        """
        self.base_path = self.get_base_path()
        self.setup_paths()
        self.load_api_keys()
        self.load_oauth_credentials()
        print(self)  # This will call __repr__ and print the configuration
    
    def __repr__(self) -> str:
        """
        Return a string representation of the configuration.
        """
        return self.get_config_str()
    
    def get_base_path(self) -> str:
        """
        Determine the base path of the project.

        Returns:
            str: The root directory of the project.
        """
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def setup_paths(self) -> None:
        """
        Set up necessary directory structures for the project.
        """
        self.dataset_path = os.path.join(self.base_path, "dataset","rag")
        self.chroma_persist_directory = os.path.join(self.base_path, "chroma_persist")

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.chroma_persist_directory, exist_ok=True)
    
    def load_api_keys(self):
        """
        Load API keys from the JSON file
        Raises:
            FileNotFoundError: If the API keys file is not found.
            json.JSONDecodeError: If the API keys file is not valid JSON.
        """
        api_keys_path = os.path.join(self.base_path, 'configs',self.api_keys_file)

        if not os.path.exists(api_keys_path):
            raise FileNotFoundError(
                f"API keys file not found: {api_keys_path}."
                f"Please create a file named {self.api_keys_file} in the configs directory"
                "with the required API keys.")
        try:
            with open(api_keys_path,'r') as f:
                keys = json.load(f)
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in API keys file: {api_keys_path}")
        self.HF_token = keys.get('HF_token','')
        self.google_api_key = keys.get('google_api_key','')
        self.google_cse_id = keys.get('google_cse_id','')
        self.validate_api_keys()

    def validate_api_keys(self):
        """
        Ensure all required API keys are present.
        Raises:
            ValueError: If any required API keys are missing.
        """
        missing_keys =[]
        if not self.HF_token:
            missing_keys.append('HF_token')
        if not self.google_api_key:
            missing_keys.append('google_api_key')
        if not self.google_cse_id:
            missing_keys.append('google_cse_id')
        
        if missing_keys:
            raise ValueError(
                f"Missing required API keys in {self.api_keys_file}: {', '.join(missing_keys)}"
                f"Please ensure all required keys are present in the '{self.api_keys_file}' file."
            )
    
    def load_oauth_credentials(self):
        """
        Load OAuth credentials from the JSON file
        Raises:
            FileNotFoundError: If the OAuth credentials file is not found.
            json.JSONDecodeError: If the OAuth credentials file is not valid JSON.
        """
        oauth_credentials_path = os.path.join(self.base_path, 'configs',self.oauth_credentials_file)

        if not os.path.exists(oauth_credentials_path):
            raise FileNotFoundError(
                f"OAuth credentials file not found: {oauth_credentials_path}."
                f"Please create a file named {self.oauth_credentials_file} in the configs directory"
                "with the required OAuth credentials.")
        try:
            with open(oauth_credentials_path,'r') as f:
                keys = json.load(f)
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON in OAuth credentials file: {oauth_credentials_path}")
        self.client_id = keys.get('client_id','')
        self.client_secret = keys.get('client_secret','')
        self.auth_uri = keys.get('auth_uri','')
        self.token_uri = keys.get('token_uri','')
        self.scopes = keys.get('scopes','')
        self.redirect_uris = keys.get('redirect_uris','')
        self.validate_oauth_credentials()

    def validate_oauth_credentials(self):
        """
        Ensure all required OAuth credentials are present.
        Raises:
            ValueError: If any required OAuth credentials are missing.
        """
        missing_keys =[]
        if not self.client_id:
            missing_keys.append('client_id')
        if not self.client_secret:
            missing_keys.append('client_secret')
        if not self.auth_uri:
            missing_keys.append('auth_uri')
        if not self.token_uri:
            missing_keys.append('token_uri')
        if not self.scopes:
            missing_keys.append('scopes')
        
        if missing_keys:
            raise ValueError(
                f"Missing required OAuth credentials in {self.oauth_credentials_file}: {', '.join(missing_keys)}"
                f"Please ensure all required keys are present in the '{self.oauth_credentials_file}' file."
            )   
        
    def get_config_str(self) -> str:
        """
        Generate a string representation of the configuration
        
        Returns:
            str: A formatted string containing all non-sensitive configuration details.
        """
        return f"""
Edvisor Configuration:
Base Path: {self.base_path}
Dataset Path: {self.dataset_path}
Chroma DB Path: {self.chroma_persist_directory}
Base Model: {self.base_model}
Embedding Model: {self.embedding_model}
Max Context Length: {self.max_context_length}
Chat History Limit: {self.chat_history_limit}
API Keys File: {self.api_keys_file}
OAuth Credentials File: {self.oauth_credentials_file}

API Keys Status:
HF_token present: {'Yes' if self.HF_token else 'No'}
Google API key present: {'Yes' if self.google_api_key else 'No'}
Google CSE ID present: {'Yes' if self.google_cse_id else 'No'}

OAuth Credentials Status:
Client ID present: {'Yes' if self.client_id else 'No'}
Client Secret present: {'Yes' if self.client_secret else 'No'}
Auth URI present: {'Yes' if self.auth_uri else 'No'}
Token URI present: {'Yes' if self.token_uri else 'No'}

Quantization Config:
Load in 4-bit: {self.quant_config.load_in_4bit}
Use double quantization: {self.quant_config.bnb_4bit_use_double_quant}
Quantization type: {self.quant_config.bnb_4bit_quant_type}
Compute dtype: {self.quant_config.bnb_4bit_compute_dtype}
"""
    
