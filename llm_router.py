"""
LLM router using LiteLLM for multi-model support.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import litellm
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRouter:
    """Router for different LLM providers using LiteLLM."""
    
    def __init__(self):
        # Set up LiteLLM configuration
        self.setup_litellm()
        
        # Available models configuration
        self.available_models = self.get_available_models()
    
    def setup_litellm(self):
        """Configure LiteLLM settings."""
        # Set logging level for LiteLLM
        litellm.set_verbose = False
        
        # Configure API keys from environment variables
        if os.getenv("OPENAI_API_KEY"):
            litellm.openai_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            litellm.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if os.getenv("COHERE_API_KEY"):
            litellm.cohere_key = os.getenv("COHERE_API_KEY")
        
        if os.getenv("MISTRAL_API_KEY"):
            litellm.mistral_key = os.getenv("MISTRAL_API_KEY")
        
        if os.getenv("GROQ_API_KEY"):
            litellm.groq_key = os.getenv("GROQ_API_KEY")
            # Also set the environment variable that LiteLLM checks directly
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """Get available models organized by provider."""
        models = {
            "OpenAI": [
                {"name": "gpt-4", "display_name": "GPT-4"},
                {"name": "gpt-4-turbo", "display_name": "GPT-4 Turbo"},
                {"name": "gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo"},
                {"name": "gpt-4o", "display_name": "GPT-4o"},
                {"name": "gpt-4o-mini", "display_name": "GPT-4o Mini"}
            ],
            "Anthropic": [
                {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus"},
                {"name": "claude-3-sonnet-20240229", "display_name": "Claude 3 Sonnet"},
                {"name": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku"},
                {"name": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"}
            ],
            "Cohere": [
                {"name": "command-r", "display_name": "Command R"},
                {"name": "command-r-plus", "display_name": "Command R+"}
            ],
            "Mistral": [
                {"name": "mistral-large-latest", "display_name": "Mistral Large"},
                {"name": "mistral-medium-latest", "display_name": "Mistral Medium"},
                {"name": "mistral-small-latest", "display_name": "Mistral Small"}
            ],
            "Groq": [
                {"name": "llama-3.3-70b-versatile", "display_name": "Llama 3.3 70B Versatile"},
                {"name": "llama-3.1-8b-instant", "display_name": "Llama 3.1 8B Instant"},
                {"name": "meta-llama/llama-4-scout-17b-16e-instruct", "display_name": "Llama 4 Scout 17B"},
                {"name": "meta-llama/llama-4-maverick-17b-128e-instruct", "display_name": "Llama 4 Maverick 17B"},
                {"name": "meta-llama/llama-guard-4-12b", "display_name": "Llama Guard 4 12B"},
                {"name": "deepseek-r1-distill-llama-70b", "display_name": "DeepSeek R1 Distill Llama 70B"},
                {"name": "deepseek-r1-distill-qwen-32b", "display_name": "DeepSeek R1 Distill Qwen 32B"},
                {"name": "qwen-qwq-32b", "display_name": "Qwen QwQ 32B"},
                {"name": "gemma2-9b-it", "display_name": "Gemma 2 9B"}
            ]
        }
        return models
    
    def get_model_list_for_ui(self) -> List[str]:
        """Get flattened list of models for UI dropdown."""
        model_list = []
        for provider, models in self.available_models.items():
            for model in models:
                model_list.append(f"{provider}: {model['display_name']}")
        return model_list
    
    def get_model_name_from_display(self, display_name: str) -> str:
        """Convert display name back to actual model name."""
        for provider, models in self.available_models.items():
            for model in models:
                if f"{provider}: {model['display_name']}" == display_name:
                    return model['name']
        return "gpt-3.5-turbo"  # Default fallback
    
    def get_provider_for_model(self, model_name: str) -> str:
        """Get the provider for a given model name."""
        for provider, models in self.available_models.items():
            for model in models:
                if model['name'] == model_name:
                    return provider.lower()
        return None
    
    def format_model_for_litellm(self, model_name: str) -> str:
        """Format model name with provider prefix for LiteLLM."""
        provider = self.get_provider_for_model(model_name)
        
        # Special case handling for specific models
        if "llama-4" in model_name or "meta-llama/llama-4" in model_name:
            return f"groq/{model_name}"
            
        if provider:
            # Special handling for different providers
            if provider == "openai":
                return model_name  # OpenAI models don't need prefix
            elif provider == "anthropic":
                return f"anthropic/{model_name}"
            elif provider == "cohere":
                return f"cohere/{model_name}"
            elif provider == "mistral":
                return f"mistral/{model_name}"
            elif provider == "groq":
                return f"groq/{model_name}"  # Always prefix Groq models
                
        return model_name  # Fallback to original name
    
    def generate_response(self, 
                         model: str, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         stream: bool = False) -> str:
        """Generate response using specified model."""
        try:
            # Format model name with provider prefix for LiteLLM
            formatted_model = self.format_model_for_litellm(model)
            logger.info(f"Using model: {formatted_model} (original: {model})")
            
            # Log available API keys (without revealing the actual keys)
            api_keys = {
                "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
                "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
                "Cohere": bool(os.getenv("COHERE_API_KEY")),
                "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
                "Groq": bool(os.getenv("GROQ_API_KEY"))
            }
            logger.info(f"Available API keys: {api_keys}")
            
            # Ensure the provider's API key is set
            provider = self.get_provider_for_model(model)
            if provider and provider == "groq" and not os.getenv("GROQ_API_KEY"):
                raise ValueError("Groq API key is required but not set")
            
            response = completion(
                model=formatted_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response  # Return generator for streaming
            else:
                return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response with model {model}: {e}")
            if "provider" in str(e).lower():
                logger.error(f"Provider issue detected. Formatted model was: {formatted_model}")
                logger.error(f"Try setting the model with explicit provider, e.g., 'groq/llama-3.1-8b-instant'")
            raise
    
    def generate_rag_response(self, 
                            model: str,
                            query: str, 
                            context_chunks: List[Dict[str, Any]], 
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> str:
        """Generate RAG response using retrieved context."""
        try:
            # Prepare context from chunks
            context_text = self._prepare_context(context_chunks)
            
            # Create system message with RAG instructions
            system_message = self._create_rag_system_message()
            
            # Create user message with context and query
            user_message = self._create_rag_user_message(context_text, query)
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            return self.generate_response(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            raise
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Document {i}: {chunk['file_name']}]\n{chunk['text']}\n"
            )
        return "\n".join(context_parts)
    
    def _create_rag_system_message(self) -> str:
        """Create system message for RAG."""
        return """You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite the relevant document names when possible
4. Provide accurate, helpful, and concise responses
5. If multiple documents contain relevant information, synthesize the information appropriately
6. Maintain a professional and informative tone"""
    
    def _create_rag_user_message(self, context: str, query: str) -> str:
        """Create user message with context and query."""
        return f"""Context from PDF documents:
{context}

Question: {query}

Please answer the question based on the provided context."""
    
    def test_model_availability(self, model: str) -> bool:
        """Test if a model is available and working."""
        try:
            # Format model name with provider prefix for LiteLLM
            formatted_model = self.format_model_for_litellm(model)
            test_messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = completion(
                model=formatted_model,
                messages=test_messages,
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.warning(f"Model {model} not available: {e}")
            return False
    
    def get_working_models(self) -> List[str]:
        """Get list of models that are currently working."""
        working_models = []
        for provider, models in self.available_models.items():
            for model in models:
                if self.test_model_availability(model['name']):
                    working_models.append(f"{provider}: {model['display_name']}")
        return working_models

class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def get_conversation_messages(self, system_message: str = None) -> List[Dict[str, str]]:
        """Get formatted conversation messages."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.extend(self.conversation_history)
        return messages
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Utility functions
def get_default_model() -> str:
    """Get default model based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-3.5-turbo"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "claude-3-haiku-20240307"
    elif os.getenv("COHERE_API_KEY"):
        return "command-r"
    elif os.getenv("GROQ_API_KEY"):
        return "llama-3.1-8b-instant"
    else:
        return "gpt-3.5-turbo"  # Default fallback

def validate_api_keys() -> Dict[str, bool]:
    """Validate which API keys are available."""
    return {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Cohere": bool(os.getenv("COHERE_API_KEY")),
        "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "Groq": bool(os.getenv("GROQ_API_KEY"))
    } 