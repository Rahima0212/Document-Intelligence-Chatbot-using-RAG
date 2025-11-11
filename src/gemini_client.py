# gemini_client.py

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
import numpy as np
import requests

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_LLM_MODEL = os.getenv('GEMINI_LLM_MODEL', 'gemini-2.5-flash')
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')

class GeminiClient:
    """Official Google Generative AI (Gemini) client wrapper."""
    
    def __init__(self, api_key: str = None, debug: bool = False):
        """Create a GeminiClient.

        Args:
            api_key: optional API key override. If not provided, reads from env var `GEMINI_API_KEY`.
            debug: if True, the client will expose debug information and print a masked status on init.
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.debug = bool(debug)
        self.available = bool(self.api_key)

        if self.available:
            try:
                # The client gets the API key from the environment variable
                self.client = genai.Client(api_key=self.api_key)
                self._models_ok = True
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")
                self.available = False
                self._models_ok = False
        else:
            print("⚠️ Gemini API key not found. Gemini features will be unavailable.")
            self._models_ok = False

        if self.debug:
            # provide a short masked status printout for debugging (does not expose the key)
            self.print_debug_status()

    def _mask_key(self, key: str) -> str:
        """Return a masked version of the API key for safe debug output."""
        if not key:
            return '<missing>'
        if len(key) <= 8:
            return key[0:1] + '*' * (len(key) - 1)
        return key[0:4] + '*' * (len(key) - 8) + key[-4:]

    def get_status(self) -> Dict[str, Any]:
        """Return a dictionary summarizing client configuration (safe to log).

        Note: this does not reveal the full API key, only a masked version.
        """
        return {
            'available': self.available,
            'api_key_set': bool(self.api_key),
            'api_key_masked': self._mask_key(self.api_key) if self.api_key else '<missing>',
            'llm_model': GEMINI_LLM_MODEL,
            'embedding_model': GEMINI_EMBEDDING_MODEL,
            'models_initialized': getattr(self, '_models_ok', False)
        }

    def print_debug_status(self) -> None:
        """Print a concise, non-sensitive debug summary to stdout."""
        status = self.get_status()
        print("GeminiClient status:")
        print(f"  available: {status['available']}")
        print(f"  api_key_set: {status['api_key_set']}")
        print(f"  api_key_masked: {status['api_key_masked']}")
        print(f"  llm_model: {status['llm_model']}")
        print(f"  embedding_model: {status['embedding_model']}")
        print(f"  models_initialized: {status['models_initialized']}")
    
    def _init_embedding_model(self):
        """Initialize the embedding model with retries."""
        try:
            self.embedding_model = genai.get_model(GEMINI_EMBEDDING_MODEL)
        except Exception as e:
            print(f"Warning: Could not initialize embedding model: {e}")
            self.embedding_model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini's embedding model."""
        if not self.available:
            raise RuntimeError('Gemini client not configured (missing API key)')
        
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text
            )
            embeddings.append(result.embeddings)
        return embeddings

    def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Generate chat completion using Gemini model."""
        if not self.available:
            raise RuntimeError('Gemini client not configured (missing API key)')
        
        # Combine all messages into a single prompt for now
        # Since gemini-2.5-flash is primarily for single-turn interactions
        combined_prompt = ""
        
        if system_prompt:
            combined_prompt += f"{system_prompt}\n\n"
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                combined_prompt += f"User: {content}\n"
            elif role == 'assistant':
                combined_prompt += f"Assistant: {content}\n"
            # Skip system messages as they're handled above
        
        # Generate response using the flash model
        response = self.client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=combined_prompt.strip()
        )
        
        # Convert response to expected format
        return {
            "choices": [{
                "message": {
                    "content": response.text,
                    "role": "assistant"
                }
            }]
        }
    def generate_chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 512
    ):
        """Stream a chat completion from Gemini using generate_content_stream.

        Yields chunk objects (or their text) as they arrive. Caller should
        concatenate chunk.text to build the full response.
        """
        if not self.available:
            raise RuntimeError('Gemini client not configured (missing API key)')

        # Build a combined prompt similar to generate_chat_completion
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n\n"

        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                combined_prompt += f"User: {content}\n"
            elif role == 'assistant':
                combined_prompt += f"Assistant: {content}\n"

        # Use the streaming API from the official client
        try:
            stream = self.client.models.generate_content_stream(
                model=GEMINI_LLM_MODEL,
                contents=combined_prompt.strip()
            )
        except Exception as e:
            # Re-raise with helpful context
            raise RuntimeError(f"Failed to start Gemini streaming generation: {e}")

        # The iterator yields chunk-like objects; yield them to the caller
        for chunk in stream:
            # Some SDK versions provide `.text`, others `.delta` etc.
            # Expose the chunk object and also its text attribute if present.
            text = getattr(chunk, 'text', None) or getattr(chunk, 'delta', None) or None
            yield {
                'chunk': chunk,
                'text': text
            }