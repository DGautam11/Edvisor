# src/model.py

from src.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    """
    Model class for managing the language model and tokenizer in the Edvisor project.

    This class is responsible for initializing and storing the language model
    and tokenizer based on the configuration settings. It provides methods to
    set up the tokenizer and model, and to retrieve them for use in other parts
    of the application.

    Attributes:
        config (Config): Configuration object containing model settings.
        tokenizer (AutoTokenizer): The tokenizer for the language model.
        model (AutoModelForCausalLM): The main language model.

    Methods:
        set_tokenizer(): Initializes and returns the tokenizer.
        set_model(): Initializes and returns the language model.
        get_model_tokenizer(): Returns the model and tokenizer as a tuple.
    """

    def __init__(self):
        """
        Initialize the Model instance.

        Sets up the configuration and initializes the tokenizer and model.
        """
        self.config = Config()  # Load configuration
        self.tokenizer = self.set_tokenizer()  # Initialize tokenizer
        self.model = self.set_model()  # Initialize model

    def set_tokenizer(self):
        """
        Initialize and return the tokenizer.

        Returns:
            AutoTokenizer: The initialized tokenizer for the language model.
        """
        # Load the tokenizer using the base model name from config
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            token=self.config.HF_token  # Use the HuggingFace token for authentication
        )
        return tokenizer

    def set_model(self):
        """
        Initialize and return the language model.

        Returns:
            AutoModelForCausalLM: The initialized language model.
        """
        # Load the model using configuration settings
        return AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            device_map={"": 0},  # Map model to first available device (usually GPU)
            quantization_config=self.config.quant_config,  # Apply quantization settings
            token=self.config.HF_token  # Use the HuggingFace token for authentication
        )

    def get_model_tokenizer(self):
        """
        Retrieve the model and tokenizer.

        Returns:
            tuple: A tuple containing the model and tokenizer (model, tokenizer).
        """
        return self.model, self.tokenizer