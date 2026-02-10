"""Hugging Face model loading and inference."""

import gc
import threading
from typing import Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import settings
from utils.logger import setup_logger
from model.prompts import PromptBuilder
from model.reasoning import ReasoningExtractor
from model.streaming import StreamHandler, TokenBuffer

logger = setup_logger(__name__)


class ModelHandler:
    """Handle Hugging Face model loading and inference."""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_memory: Optional[int] = None
    ):
        """Initialize model handler.
        
        Args:
            model_name: HuggingFace model name.
            device: Device to use (cpu/cuda).
            max_memory: Maximum memory in MB.
        """
        self.model_name = model_name or settings.MODEL_NAME
        self.device = device or settings.DEVICE
        self.max_memory = max_memory or settings.MAX_MEMORY_MB
        
        self._model = None
        self._tokenizer = None
        self._prompt_builder = PromptBuilder()
        self._reasoning_extractor = ReasoningExtractor()
        self._stream_handler = StreamHandler()
        
        logger.info(f"ModelHandler initialized: {self.model_name}")
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Set memory limit if specified
            if self.max_memory:
                # Convert MB to bytes for transformers
                max_memory_dict = {0: f"{self.max_memory}MB"}
            else:
                max_memory_dict = None
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.CACHE_DIR,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model with optimizations for CPU
            load_kwargs = {
                "cache_dir": settings.CACHE_DIR,
                "trust_remote_code": True,
                "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
            }
            
            if max_memory_dict:
                load_kwargs["max_memory"] = max_memory_dict
            
            if self.device == "cpu":
                load_kwargs["low_cpu_mem_usage"] = True
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            self._model.to(self.device)
            self._model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = None,
        max_new_tokens: int = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """Generate text with streaming.
        
        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_new_tokens: Maximum new tokens to generate.
            stop_sequences: Optional stop sequences.
            
        Yields:
            Generated tokens.
        """
        self._load_model()
        
        temperature = temperature or settings.TEMPERATURE
        max_new_tokens = max_new_tokens or settings.MAX_TOKENS
        
        try:
            # Tokenize input
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            
            # Set up streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation kwargs
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": settings.TOP_P,
                "top_k": settings.TOP_K,
                "do_sample": temperature > 0.0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "streamer": streamer,
            }
            
            # Generate in separate thread
            thread = threading.Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            generated_text = ""
            for text in streamer:
                if self._stream_handler.should_stop:
                    break
                
                generated_text += text
                
                # Check stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            # Truncate at stop sequence
                            idx = generated_text.find(stop_seq)
                            if idx > 0:
                                remaining = generated_text[:idx]
                                if len(remaining) > len(text) - len(generated_text) + len(remaining):
                                    yield remaining[len(generated_text) - len(text):idx]
                            return
                
                yield text
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_new_tokens: int = None
    ) -> str:
        """Generate text without streaming.
        
        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_new_tokens: Maximum new tokens.
            
        Returns:
            Generated text.
        """
        self._load_model()
        
        temperature = temperature or settings.TEMPERATURE
        max_new_tokens = max_new_tokens or settings.MAX_TOKENS
        
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=settings.TOP_P,
                    top_k= settings.TOP_K,
                    do_sample=temperature > 0.0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def extract_reasoning(self, text: str) -> Tuple[str, str]:
        """Extract reasoning and answer from model output.
        
        Args:
            text: Model output text.
            
        Returns:
            Tuple of (reasoning, answer).
        """
        return self._reasoning_extractor.extract(text)
    
    def get_stream_handler(self) -> StreamHandler:
        """Get the stream handler for controlling generation."""
        return self._stream_handler
    
    def stop_generation(self):
        """Stop ongoing generation."""
        self._stream_handler.stop()
    
    def clear_memory(self):
        """Clear model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cleared from memory")
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._model is not None,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else None,
        }
