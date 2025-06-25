# Hugging Face Integration Guide

Integrate PRSM with Hugging Face's ecosystem for access to thousands of pre-trained models, datasets, and AI tools.

## 🎯 Overview

This guide covers integrating PRSM with Hugging Face Hub, Transformers library, Datasets, and Inference API for comprehensive AI model access and deployment.

## 📋 Prerequisites

- PRSM instance configured
- Python 3.8+ installed
- Hugging Face account (optional for public models)
- GPU support (recommended for large models)

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install Hugging Face libraries
pip install transformers
pip install datasets
pip install huggingface_hub
pip install accelerate
pip install bitsandbytes  # For quantization
pip install torch torchvision torchaudio  # PyTorch

# For specialized tasks
pip install sentence-transformers  # Sentence embeddings
pip install diffusers  # Image generation
pip install tokenizers  # Fast tokenizers
```

### 2. Basic Configuration

```python
# prsm/integrations/huggingface/config.py
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings
from huggingface_hub import login

class HuggingFaceConfig(BaseSettings):
    """Hugging Face integration configuration."""
    
    # Authentication
    hf_token: Optional[str] = None
    hf_cache_dir: str = "./cache/huggingface"
    
    # Model configurations
    default_text_model: str = "microsoft/DialoGPT-medium"
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_classification_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    default_generation_model: str = "microsoft/DialoGPT-medium"
    
    # Hardware configurations
    device: str = "auto"  # auto, cpu, cuda
    use_gpu: bool = True
    max_memory_gb: Optional[int] = None
    use_quantization: bool = False
    
    # Generation parameters
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Batch processing
    batch_size: int = 8
    max_concurrent_requests: int = 4
    
    class Config:
        env_prefix = "PRSM_HF_"

# Global configuration
hf_config = HuggingFaceConfig()

# Authenticate with Hugging Face Hub
if hf_config.hf_token:
    login(token=hf_config.hf_token)
```

### 3. Basic Model Manager

```python
# prsm/integrations/huggingface/model_manager.py
import asyncio
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    AutoModelForSequenceClassification, pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import gc

from prsm.integrations.huggingface.config import hf_config

logger = logging.getLogger(__name__)

class HuggingFaceModelManager:
    """Manages Hugging Face models for PRSM."""
    
    def __init__(self):
        self.config = hf_config
        self.loaded_models: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best device for inference."""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available() and self.config.use_gpu:
            return "cuda"
        elif torch.backends.mps.is_available() and self.config.use_gpu:
            return "mps"
        else:
            return "cpu"
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled."""
        if not self.config.use_quantization or self.device == "cpu":
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    async def load_model(
        self,
        model_name: str,
        model_type: str = "causal",
        use_cache: bool = True
    ) -> Any:
        """Load a Hugging Face model."""
        if use_cache and model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.hf_cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on type
            quantization_config = self._get_quantization_config()
            
            if model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.config.hf_cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            elif model_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.config.hf_cache_dir,
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.config.hf_cache_dir,
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            
            # Move to device if not using device_map
            if not quantization_config and self.device != "cuda":
                model = model.to(self.device)
            
            model_info = {
                "model": model,
                "tokenizer": tokenizer,
                "model_type": model_type,
                "model_name": model_name
            }
            
            if use_cache:
                self.loaded_models[model_name] = model_info
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def load_pipeline(
        self,
        task: str,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Any:
        """Load a Hugging Face pipeline."""
        cache_key = f"{task}_{model_name or 'default'}"
        
        if use_cache and cache_key in self.pipelines:
            return self.pipelines[cache_key]
        
        try:
            logger.info(f"Loading pipeline for task: {task}")
            
            pipeline_kwargs = {
                "task": task,
                "device": 0 if self.device == "cuda" else -1,
                "model_kwargs": {"cache_dir": self.config.hf_cache_dir}
            }
            
            if model_name:
                pipeline_kwargs["model"] = model_name
            
            # Create pipeline
            pipe = pipeline(**pipeline_kwargs)
            
            if use_cache:
                self.pipelines[cache_key] = pipe
            
            logger.info(f"Successfully loaded pipeline: {task}")
            return pipe
            
        except Exception as e:
            logger.error(f"Failed to load pipeline {task}: {e}")
            raise
    
    async def load_sentence_transformer(
        self,
        model_name: Optional[str] = None,
        use_cache: bool = True
    ) -> SentenceTransformer:
        """Load a sentence transformer model."""
        model_name = model_name or self.config.default_embedding_model
        cache_key = f"sentence_transformer_{model_name}"
        
        if use_cache and cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            logger.info(f"Loading sentence transformer: {model_name}")
            
            model = SentenceTransformer(
                model_name,
                cache_folder=self.config.hf_cache_dir,
                device=self.device
            )
            
            if use_cache:
                self.loaded_models[cache_key] = model
            
            logger.info(f"Successfully loaded sentence transformer: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys()) + list(self.pipelines.keys())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["cuda"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        
        return {
            "loaded_models": len(self.loaded_models),
            "loaded_pipelines": len(self.pipelines),
            "memory": memory_info
        }

# Global model manager
hf_model_manager = HuggingFaceModelManager()
```

## 🤖 Text Generation Integration

### 1. Conversational AI

```python
# prsm/integrations/huggingface/generation.py
import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import torch
from transformers import GenerationConfig

from prsm.integrations.huggingface.model_manager import hf_model_manager
from prsm.integrations.huggingface.config import hf_config
from prsm.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class HuggingFaceTextGenerator:
    """Text generation using Hugging Face models."""
    
    def __init__(self):
        self.config = hf_config
        self.model_manager = hf_model_manager
        
    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """Generate text response using Hugging Face model."""
        try:
            model_name = model_name or self.config.default_generation_model
            
            # Load model
            model_info = await self.model_manager.load_model(
                model_name, 
                model_type="causal"
            )
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model_manager.device)
            
            # Generation configuration
            generation_config = GenerationConfig(
                max_length=max_length or self.config.max_length,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                top_k=top_k or self.config.top_k,
                repetition_penalty=repetition_penalty or self.config.repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    generation_config=generation_config,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def generate_streaming(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text response."""
        try:
            model_name = model_name or self.config.default_generation_model
            
            # Load model
            model_info = await self.model_manager.load_model(
                model_name,
                model_type="causal"
            )
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.model_manager.device)
            
            # Generate token by token
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / kwargs.get("temperature", 0.8)
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Check for EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Decode and yield partial response
                    partial_text = tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                    
                    yield partial_text
                    
                    # Update inputs for next iteration
                    inputs = torch.cat([inputs, next_token], dim=-1)
                    
                    # Yield control to event loop
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query using Hugging Face models."""
        try:
            # Generate response
            response_text = await self.generate_response(
                prompt=request.prompt,
                max_length=request.max_tokens,
                temperature=request.temperature
            )
            
            # Create response
            response = QueryResponse(
                query_id=f"hf_{request.user_id}_{int(asyncio.get_event_loop().time())}",
                final_answer=response_text,
                user_id=request.user_id,
                model=self.config.default_generation_model,
                metadata={
                    "huggingface_integration": True,
                    "device": self.model_manager.device
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

# Global text generator
hf_text_generator = HuggingFaceTextGenerator()
```

### 2. Conversational Pipeline

```python
# prsm/integrations/huggingface/conversation.py
import asyncio
from typing import Dict, Any, List, Optional
from transformers import Conversation

from prsm.integrations.huggingface.model_manager import hf_model_manager

class HuggingFaceConversationManager:
    """Manage conversations using Hugging Face conversational pipeline."""
    
    def __init__(self):
        self.model_manager = hf_model_manager
        self.conversations: Dict[str, Conversation] = {}
        
    async def start_conversation(
        self,
        user_id: str,
        initial_message: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> str:
        """Start a new conversation."""
        try:
            # Load conversational pipeline
            pipeline = await self.model_manager.load_pipeline(
                "conversational",
                model_name=model_name
            )
            
            # Create conversation
            conversation = Conversation(text=initial_message) if initial_message else Conversation()
            
            if initial_message:
                # Generate initial response
                result = pipeline(conversation)
                response = result.generated_responses[-1]
            else:
                response = "Hello! How can I help you today?"
            
            # Store conversation
            self.conversations[user_id] = conversation
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise
    
    async def continue_conversation(
        self,
        user_id: str,
        message: str,
        model_name: Optional[str] = None
    ) -> str:
        """Continue an existing conversation."""
        try:
            # Get or create conversation
            if user_id not in self.conversations:
                await self.start_conversation(user_id)
            
            conversation = self.conversations[user_id]
            
            # Load pipeline
            pipeline = await self.model_manager.load_pipeline(
                "conversational",
                model_name=model_name
            )
            
            # Add user message
            conversation.add_user_input(message)
            
            # Generate response
            result = pipeline(conversation)
            response = result.generated_responses[-1]
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to continue conversation: {e}")
            raise
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a user."""
        if user_id not in self.conversations:
            return []
        
        conversation = self.conversations[user_id]
        history = []
        
        # Format conversation history
        for i, (user_input, bot_response) in enumerate(
            zip(conversation.past_user_inputs, conversation.generated_responses)
        ):
            history.append({"user": user_input, "assistant": bot_response})
        
        return history
    
    def clear_conversation(self, user_id: str):
        """Clear conversation for a user."""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation user IDs."""
        return list(self.conversations.keys())

# Global conversation manager
hf_conversation_manager = HuggingFaceConversationManager()
```

## 🔍 Embeddings and Similarity

### 1. Sentence Embeddings

```python
# prsm/integrations/huggingface/embeddings.py
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
import torch

from prsm.integrations.huggingface.model_manager import hf_model_manager
from prsm.integrations.huggingface.config import hf_config

class HuggingFaceEmbeddings:
    """Generate embeddings using Hugging Face models."""
    
    def __init__(self):
        self.config = hf_config
        self.model_manager = hf_model_manager
        
    async def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Create embeddings for text(s)."""
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False
            
            model_name = model_name or self.config.default_embedding_model
            batch_size = batch_size or self.config.batch_size
            
            # Load sentence transformer
            model = await self.model_manager.load_sentence_transformer(model_name)
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = model.encode(
                    batch_texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )
                
                all_embeddings.extend(batch_embeddings)
                
                # Yield control to event loop
                await asyncio.sleep(0)
            
            # Convert to numpy array
            embeddings = np.array(all_embeddings)
            
            # Return single embedding if single text
            if single_text:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    async def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        model_name: Optional[str] = None
    ) -> float:
        """Compute semantic similarity between two texts."""
        try:
            # Get embeddings if needed
            if isinstance(text1, str):
                embedding1 = await self.create_embeddings(text1, model_name)
            else:
                embedding1 = text1
            
            if isinstance(text2, str):
                embedding2 = await self.create_embeddings(text2, model_name)
            else:
                embedding2 = text2
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise
    
    async def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        model_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar texts to a query."""
        try:
            # Get query embedding
            query_embedding = await self.create_embeddings(query_text, model_name)
            
            # Get candidate embeddings
            candidate_embeddings = await self.create_embeddings(
                candidate_texts,
                model_name
            )
            
            # Compute similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding, candidate_embedding)
                similarities.append({
                    "text": candidate_texts[i],
                    "similarity": float(similarity),
                    "index": i
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similar text search failed: {e}")
            raise
    
    async def cluster_texts(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Cluster texts based on semantic similarity."""
        try:
            from sklearn.cluster import KMeans
            
            # Get embeddings
            embeddings = await self.create_embeddings(texts, model_name)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "text": texts[i],
                    "index": i
                })
            
            return {
                "clusters": clusters,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "n_clusters": n_clusters
            }
            
        except Exception as e:
            logger.error(f"Text clustering failed: {e}")
            raise

# Global embeddings handler
hf_embeddings = HuggingFaceEmbeddings()
```

## 🏷️ Classification and Analysis

### 1. Text Classification

```python
# prsm/integrations/huggingface/classification.py
import asyncio
from typing import Dict, Any, List, Optional, Union
from transformers import pipeline

from prsm.integrations.huggingface.model_manager import hf_model_manager
from prsm.integrations.huggingface.config import hf_config

class HuggingFaceClassifier:
    """Text classification using Hugging Face models."""
    
    def __init__(self):
        self.config = hf_config
        self.model_manager = hf_model_manager
    
    async def classify_sentiment(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify sentiment of text(s)."""
        try:
            model_name = model_name or self.config.default_classification_model
            
            # Load sentiment analysis pipeline
            classifier = await self.model_manager.load_pipeline(
                "sentiment-analysis",
                model_name=model_name
            )
            
            # Classify
            results = classifier(texts)
            
            # Format results
            if isinstance(texts, str):
                return {
                    "text": texts,
                    "sentiment": results[0]["label"],
                    "confidence": results[0]["score"]
                }
            else:
                formatted_results = []
                for i, result in enumerate(results):
                    formatted_results.append({
                        "text": texts[i],
                        "sentiment": result["label"],
                        "confidence": result["score"]
                    })
                return formatted_results
                
        except Exception as e:
            logger.error(f"Sentiment classification failed: {e}")
            raise
    
    async def classify_topic(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify topic/category of text(s)."""
        try:
            if labels:
                # Zero-shot classification
                classifier = await self.model_manager.load_pipeline(
                    "zero-shot-classification",
                    model_name=model_name
                )
                
                results = classifier(texts, labels)
                
                # Format results
                if isinstance(texts, str):
                    return {
                        "text": texts,
                        "predictions": [
                            {"label": label, "score": score}
                            for label, score in zip(results["labels"], results["scores"])
                        ],
                        "top_prediction": {
                            "label": results["labels"][0],
                            "score": results["scores"][0]
                        }
                    }
                else:
                    formatted_results = []
                    for i, result in enumerate(results):
                        formatted_results.append({
                            "text": texts[i],
                            "predictions": [
                                {"label": label, "score": score}
                                for label, score in zip(result["labels"], result["scores"])
                            ],
                            "top_prediction": {
                                "label": result["labels"][0],
                                "score": result["scores"][0]
                            }
                        })
                    return formatted_results
            else:
                # Regular text classification
                classifier = await self.model_manager.load_pipeline(
                    "text-classification",
                    model_name=model_name
                )
                
                results = classifier(texts)
                
                if isinstance(texts, str):
                    return {
                        "text": texts,
                        "category": results[0]["label"],
                        "confidence": results[0]["score"]
                    }
                else:
                    formatted_results = []
                    for i, result in enumerate(results):
                        formatted_results.append({
                            "text": texts[i],
                            "category": result["label"],
                            "confidence": result["score"]
                        })
                    return formatted_results
                    
        except Exception as e:
            logger.error(f"Topic classification failed: {e}")
            raise
    
    async def extract_entities(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract named entities from text(s)."""
        try:
            # Load NER pipeline
            ner = await self.model_manager.load_pipeline(
                "ner",
                model_name=model_name,
                aggregation_strategy="simple"
            )
            
            results = ner(texts)
            
            # Format results
            if isinstance(texts, str):
                entities = []
                for entity in results:
                    entities.append({
                        "text": entity["word"],
                        "label": entity["entity_group"],
                        "confidence": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"]
                    })
                
                return {
                    "text": texts,
                    "entities": entities
                }
            else:
                formatted_results = []
                for i, result in enumerate(results):
                    entities = []
                    for entity in result:
                        entities.append({
                            "text": entity["word"],
                            "label": entity["entity_group"],
                            "confidence": entity["score"],
                            "start": entity["start"],
                            "end": entity["end"]
                        })
                    
                    formatted_results.append({
                        "text": texts[i],
                        "entities": entities
                    })
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise
    
    async def summarize_text(
        self,
        texts: Union[str, List[str]],
        model_name: Optional[str] = None,
        max_length: int = 150,
        min_length: int = 50
    ) -> Union[str, List[str]]:
        """Summarize text(s)."""
        try:
            # Load summarization pipeline
            summarizer = await self.model_manager.load_pipeline(
                "summarization",
                model_name=model_name
            )
            
            # Summarize
            results = summarizer(
                texts,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            # Extract summaries
            if isinstance(texts, str):
                return results[0]["summary_text"]
            else:
                return [result["summary_text"] for result in results]
                
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            raise

# Global classifier
hf_classifier = HuggingFaceClassifier()
```

## 🖼️ Multimodal Integration

### 1. Image and Vision Tasks

```python
# prsm/integrations/huggingface/vision.py
import asyncio
import base64
import io
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import requests

from prsm.integrations.huggingface.model_manager import hf_model_manager

class HuggingFaceVision:
    """Computer vision tasks using Hugging Face models."""
    
    def __init__(self):
        self.model_manager = hf_model_manager
    
    async def classify_image(
        self,
        image: Union[str, Image.Image],
        model_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Classify image content."""
        try:
            # Load image classification pipeline
            classifier = await self.model_manager.load_pipeline(
                "image-classification",
                model_name=model_name
            )
            
            # Process image
            if isinstance(image, str):
                if image.startswith("http"):
                    # Download image
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content))
                elif image.startswith("data:image"):
                    # Base64 encoded image
                    image_data = base64.b64decode(image.split(",")[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image)
            
            # Classify
            results = classifier(image, top_k=top_k)
            
            return [
                {
                    "label": result["label"],
                    "confidence": result["score"]
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            raise
    
    async def generate_image_caption(
        self,
        image: Union[str, Image.Image],
        model_name: Optional[str] = None
    ) -> str:
        """Generate caption for image."""
        try:
            # Load image captioning pipeline
            captioner = await self.model_manager.load_pipeline(
                "image-to-text",
                model_name=model_name
            )
            
            # Process image (similar to classify_image)
            if isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content))
                elif image.startswith("data:image"):
                    image_data = base64.b64decode(image.split(",")[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = Image.open(image)
            
            # Generate caption
            result = captioner(image)
            
            if isinstance(result, list):
                return result[0]["generated_text"]
            else:
                return result["generated_text"]
                
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            raise
    
    async def answer_visual_question(
        self,
        image: Union[str, Image.Image],
        question: str,
        model_name: Optional[str] = None
    ) -> str:
        """Answer questions about images."""
        try:
            # Load VQA pipeline
            vqa = await self.model_manager.load_pipeline(
                "visual-question-answering",
                model_name=model_name
            )
            
            # Process image
            if isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content))
                elif image.startswith("data:image"):
                    image_data = base64.b64decode(image.split(",")[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = Image.open(image)
            
            # Answer question
            result = vqa(image=image, question=question)
            
            return result["answer"]
            
        except Exception as e:
            logger.error(f"Visual question answering failed: {e}")
            raise
    
    async def detect_objects(
        self,
        image: Union[str, Image.Image],
        model_name: Optional[str] = None,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect objects in image."""
        try:
            # Load object detection pipeline
            detector = await self.model_manager.load_pipeline(
                "object-detection",
                model_name=model_name
            )
            
            # Process image
            if isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content))
                elif image.startswith("data:image"):
                    image_data = base64.b64decode(image.split(",")[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = Image.open(image)
            
            # Detect objects
            results = detector(image, threshold=threshold)
            
            return [
                {
                    "label": result["label"],
                    "confidence": result["score"],
                    "box": result["box"]  # {"xmin": x, "ymin": y, "xmax": x, "ymax": y}
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise

# Global vision handler
hf_vision = HuggingFaceVision()
```

## 📊 Monitoring and Analytics

### Performance Monitoring

```python
# prsm/integrations/huggingface/monitoring.py
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
import torch

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
hf_requests_total = Counter(
    'prsm_huggingface_requests_total',
    'Total Hugging Face requests',
    ['model_name', 'task', 'status']
)

hf_inference_duration = Histogram(
    'prsm_huggingface_inference_duration_seconds',
    'Hugging Face inference duration',
    ['model_name', 'task']
)

hf_model_memory_usage = Gauge(
    'prsm_huggingface_model_memory_bytes',
    'Memory usage by Hugging Face models',
    ['model_name']
)

@dataclass
class HuggingFaceMetrics:
    """Hugging Face performance metrics."""
    timestamp: datetime
    model_name: str
    task: str
    inference_time: float
    memory_usage: float
    gpu_memory_usage: Optional[float]
    input_length: int
    output_length: Optional[int]
    status: str

class HuggingFaceMonitor:
    """Monitor Hugging Face integration performance."""
    
    def __init__(self):
        self.metrics_history: List[HuggingFaceMetrics] = []
        
    def start_inference(self, model_name: str, task: str) -> Dict[str, Any]:
        """Start monitoring an inference request."""
        return {
            "start_time": time.time(),
            "model_name": model_name,
            "task": task,
            "start_memory": self._get_memory_usage(),
            "start_gpu_memory": self._get_gpu_memory_usage()
        }
    
    def end_inference(
        self,
        context: Dict[str, Any],
        input_length: int,
        output_length: Optional[int] = None,
        status: str = "success"
    ):
        """End monitoring and record metrics."""
        end_time = time.time()
        inference_time = end_time - context["start_time"]
        
        # Calculate memory usage
        current_memory = self._get_memory_usage()
        memory_usage = current_memory - context["start_memory"]
        
        current_gpu_memory = self._get_gpu_memory_usage()
        gpu_memory_usage = None
        if current_gpu_memory and context["start_gpu_memory"]:
            gpu_memory_usage = current_gpu_memory - context["start_gpu_memory"]
        
        # Record Prometheus metrics
        hf_requests_total.labels(
            model_name=context["model_name"],
            task=context["task"],
            status=status
        ).inc()
        
        hf_inference_duration.labels(
            model_name=context["model_name"],
            task=context["task"]
        ).observe(inference_time)
        
        hf_model_memory_usage.labels(
            model_name=context["model_name"]
        ).set(current_memory)
        
        # Store detailed metrics
        metrics = HuggingFaceMetrics(
            timestamp=datetime.utcnow(),
            model_name=context["model_name"],
            task=context["task"],
            inference_time=inference_time,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            input_length=input_length,
            output_length=output_length,
            status=status
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        logger.info(
            f"HF inference completed: {context['model_name']} "
            f"({context['task']}) in {inference_time:.2f}s"
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 requests
        
        # Calculate statistics
        inference_times = [m.inference_time for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        # Model usage statistics
        model_usage = {}
        task_usage = {}
        
        for metric in recent_metrics:
            model_usage[metric.model_name] = model_usage.get(metric.model_name, 0) + 1
            task_usage[metric.task] = task_usage.get(metric.task, 0) + 1
        
        return {
            "total_requests": len(recent_metrics),
            "average_inference_time": sum(inference_times) / len(inference_times),
            "max_inference_time": max(inference_times),
            "min_inference_time": min(inference_times),
            "average_memory_usage": sum(memory_usage) / len(memory_usage),
            "model_usage": model_usage,
            "task_usage": task_usage,
            "success_rate": len([m for m in recent_metrics if m.status == "success"]) / len(recent_metrics) * 100
        }

# Global monitor
hf_monitor = HuggingFaceMonitor()

# Decorator for monitoring
def monitor_hf_inference(model_name: str, task: str):
    """Decorator to monitor Hugging Face inference."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = hf_monitor.start_inference(model_name, task)
            
            try:
                result = await func(*args, **kwargs)
                
                # Try to get input/output lengths
                input_length = len(str(args[0])) if args else 0
                output_length = len(str(result)) if result else 0
                
                hf_monitor.end_inference(
                    context,
                    input_length,
                    output_length,
                    "success"
                )
                
                return result
                
            except Exception as e:
                hf_monitor.end_inference(
                    context,
                    0,
                    0,
                    "error"
                )
                raise
        
        return wrapper
    return decorator
```

## 📋 FastAPI Integration

### API Endpoints

```python
# prsm/api/huggingface_endpoints.py
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import base64

from prsm.integrations.huggingface.generation import hf_text_generator
from prsm.integrations.huggingface.embeddings import hf_embeddings
from prsm.integrations.huggingface.classification import hf_classifier
from prsm.integrations.huggingface.vision import hf_vision
from prsm.integrations.huggingface.model_manager import hf_model_manager

router = APIRouter(prefix="/api/v1/huggingface", tags=["Hugging Face"])

class GenerationRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]]
    model_name: Optional[str] = None
    normalize: bool = True

class ClassificationRequest(BaseModel):
    texts: Union[str, List[str]]
    model_name: Optional[str] = None
    labels: Optional[List[str]] = None

@router.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using Hugging Face models."""
    try:
        response = await hf_text_generator.generate_response(
            prompt=request.prompt,
            model_name=request.model_name,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for text(s)."""
    try:
        embeddings = await hf_embeddings.create_embeddings(
            texts=request.texts,
            model_name=request.model_name,
            normalize=request.normalize
        )
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify")
async def classify_text(request: ClassificationRequest):
    """Classify text using various classification models."""
    try:
        if request.labels:
            # Zero-shot classification
            result = await hf_classifier.classify_topic(
                texts=request.texts,
                model_name=request.model_name,
                labels=request.labels
            )
        else:
            # Sentiment analysis
            result = await hf_classifier.classify_sentiment(
                texts=request.texts,
                model_name=request.model_name
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vision/classify")
async def classify_image(
    image: UploadFile = File(...),
    model_name: Optional[str] = None,
    top_k: int = 5
):
    """Classify uploaded image."""
    try:
        # Read image
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content))
        
        result = await hf_vision.classify_image(
            image=pil_image,
            model_name=model_name,
            top_k=top_k
        )
        
        return {"classifications": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/loaded")
async def get_loaded_models():
    """Get list of currently loaded models."""
    models = hf_model_manager.get_loaded_models()
    memory_usage = hf_model_manager.get_memory_usage()
    
    return {
        "loaded_models": models,
        "memory_usage": memory_usage
    }

@router.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model from memory."""
    try:
        hf_model_manager.unload_model(model_name)
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

**Need help with Hugging Face integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).