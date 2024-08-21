import transformers
from model import Model

class Engine:
    """
    Engine class for managing text generation and response handling in the Edvisor chatbot.

    This class initializes the language model, sets up the text generation pipeline,
    and provides methods for formatting prompts and generating responses.

    Attributes:
        model (Model): An instance of the Model class for language model management.
        pipeline (transformers.Pipeline): The text generation pipeline.

    Methods:
        set_pipeline(): Initialize the text generation pipeline.
        format_prompt(messages): Format the conversation history into a prompt.
        get_terminators(): Get the end-of-sequence token IDs.
        generate_response(messages): Generate a response based on the conversation history.
    """

    def __init__(self):
        """Initialize the Engine with a Model instance and set up the pipeline."""
        self.model = Model()
        self.pipeline = self.set_pipeline()

    def set_pipeline(self):
        """
        Set up the text generation pipeline using the model and tokenizer.

        Returns:
            transformers.Pipeline: The initialized text generation pipeline.
        """
        model, tokenizer = self.model.get_model_tokenizer()
        return transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def format_prompt(self, messages):
        """
        Format the conversation history into a prompt for the model.

        Args:
            messages (list): List of message dictionaries in the conversation history.

        Returns:
            str: Formatted prompt string.
        """
        return self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def get_terminators(self):
        """
        Get the end-of-sequence token IDs for text generation.

        Returns:
            list: List of token IDs used as terminators.
        """
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        print(terminators)  # Debug print, consider removing in production
        return terminators

    def generate_response(self, messages):
        """
        Generate a response based on the conversation history.

        Args:
            messages (list): List of message dictionaries in the conversation history.

        Returns:
            str: Generated response text.
        """
        prompt = self.format_prompt(messages)
        terminators = self.get_terminators()
        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        # Extract the generated text, excluding the input prompt
        return outputs[0]["generated_text"][len(prompt):]