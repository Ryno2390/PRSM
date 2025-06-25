# LangChain Integration Guide

Integrate PRSM with LangChain for advanced AI agent workflows, chain operations, and sophisticated language model applications.

## 🎯 Overview

This guide covers integrating PRSM with LangChain to leverage its powerful agent framework, chain compositions, and tool integrations for building sophisticated AI applications.

## 📋 Prerequisites

- PRSM instance configured
- Python 3.9+ installed
- LangChain framework knowledge
- API keys for language models

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install LangChain and related packages
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-experimental
pip install langsmith

# Additional dependencies for specific integrations
pip install chromadb  # Vector store
pip install faiss-cpu  # Facebook AI Similarity Search
pip install tiktoken  # OpenAI tokenizer
```

### 2. Basic Configuration

```python
# prsm/integrations/langchain/config.py
import os
from typing import Optional, Dict, Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from pydantic import BaseSettings

class LangChainConfig(BaseSettings):
    """LangChain integration configuration."""
    
    # Model configurations
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Embedding configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_chunk_size: int = 1000
    embedding_chunk_overlap: int = 200
    
    # Vector store configuration
    vector_store_path: str = "./data/vector_store"
    vector_store_collection: str = "prsm_documents"
    
    # Memory configuration
    memory_max_tokens: int = 4000
    memory_return_messages: bool = True
    
    # LangSmith configuration (optional)
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "prsm-integration"
    
    class Config:
        env_prefix = "PRSM_LANGCHAIN_"

# Global configuration
langchain_config = LangChainConfig()

# LangSmith setup for monitoring (optional)
if langchain_config.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_config.langsmith_project
```

### 3. Basic LangChain Integration

```python
# prsm/integrations/langchain/basic.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler

from prsm.models.query import QueryRequest, QueryResponse
from prsm.integrations.langchain.config import langchain_config

logger = logging.getLogger(__name__)

class PRSMCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for PRSM integration."""
    
    def __init__(self):
        self.tokens_used = 0
        self.api_calls = 0
        self.execution_time = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.api_calls += 1
        logger.debug(f"LLM call started: {serialized.get('name', 'unknown')}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        if hasattr(response, 'llm_output') and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            self.tokens_used += token_usage.get('total_tokens', 0)
        logger.debug(f"LLM call completed. Total tokens: {self.tokens_used}")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors."""
        logger.error(f"LLM error: {error}")

class LangChainPRSMIntegration:
    """Main integration class for PRSM and LangChain."""
    
    def __init__(self):
        self.config = langchain_config
        self.callback_handler = PRSMCallbackHandler()
        
        # Initialize language model
        self.llm = ChatOpenAI(
            model_name=self.config.openai_model,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
            openai_api_key=self.config.openai_api_key,
            callbacks=[self.callback_handler]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            max_token_limit=self.config.memory_max_tokens
        )
        
        # Initialize conversation chain
        self.conversation_chain = self._create_conversation_chain()
    
    def _create_conversation_chain(self) -> ConversationChain:
        """Create a conversation chain with memory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are PRSM, an advanced AI assistant. You are helpful, accurate, and concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        return ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query using LangChain."""
        try:
            # Reset callback handler for this query
            self.callback_handler.tokens_used = 0
            self.callback_handler.api_calls = 0
            
            # Add user context to memory if provided
            if request.context:
                context_messages = self._parse_context(request.context)
                for message in context_messages:
                    self.memory.chat_memory.add_message(message)
            
            # Process the query
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.conversation_chain.predict,
                request.prompt
            )
            
            # Create response
            response = QueryResponse(
                query_id=f"lc_{request.user_id}_{int(asyncio.get_event_loop().time())}",
                final_answer=result,
                user_id=request.user_id,
                model=self.config.openai_model,
                token_usage=self.callback_handler.tokens_used,
                metadata={
                    "langchain_integration": True,
                    "api_calls": self.callback_handler.api_calls,
                    "memory_messages": len(self.memory.chat_memory.messages)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LangChain query processing failed: {e}")
            raise
    
    def _parse_context(self, context: str) -> List[BaseMessage]:
        """Parse context string into LangChain messages."""
        messages = []
        
        # Simple context parsing - can be enhanced for complex formats
        lines = context.split('\n')
        for line in lines:
            if line.startswith('Human: '):
                messages.append(HumanMessage(content=line[7:]))
            elif line.startswith('AI: '):
                messages.append(AIMessage(content=line[4:]))
        
        return messages
    
    def clear_memory(self, user_id: Optional[str] = None):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info(f"Memory cleared for user: {user_id}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state."""
        return {
            "message_count": len(self.memory.chat_memory.messages),
            "token_estimate": sum(len(msg.content) for msg in self.memory.chat_memory.messages) // 4,
            "last_messages": [msg.content[:100] for msg in self.memory.chat_memory.messages[-3:]]
        }

# Global integration instance
langchain_integration = LangChainPRSMIntegration()
```

## 🔗 Advanced Chain Implementations

### 1. RAG Chain with PRSM

```python
# prsm/integrations/langchain/rag_chain.py
import asyncio
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from prsm.integrations.langchain.config import langchain_config
from prsm.integrations.langchain.basic import LangChainPRSMIntegration

class PRSMRAGChain:
    """Retrieval-Augmented Generation chain for PRSM."""
    
    def __init__(self):
        self.config = langchain_config
        self.base_integration = LangChainPRSMIntegration()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.embedding_chunk_size,
            chunk_overlap=self.config.embedding_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.config.vector_store_collection,
            embedding_function=self.embeddings,
            persist_directory=self.config.vector_store_path
        )
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self) -> RetrievalQA:
        """Create the RAG chain."""
        # Custom prompt template for RAG
        prompt_template = """You are PRSM, an advanced AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.base_integration.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        try:
            # Convert to LangChain documents
            langchain_docs = []
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # Split document into chunks
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    langchain_doc = Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'document_id': doc.get('id', f"doc_{i}")
                        }
                    )
                    langchain_docs.append(langchain_doc)
            
            # Add to vector store
            doc_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_store.add_documents,
                langchain_docs
            )
            
            # Persist the vector store
            self.vector_store.persist()
            
            logger.info(f"Added {len(langchain_docs)} document chunks to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return []
    
    async def query_with_context(
        self,
        question: str,
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query with retrieval-augmented generation."""
        try:
            # Update retriever with filters if provided
            if filter_metadata:
                self.rag_chain.retriever.search_kwargs.update({
                    "filter": filter_metadata
                })
            
            # Execute the RAG chain
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.rag_chain,
                {"query": question}
            )
            
            # Format response
            response = {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "unknown")
                    }
                    for doc in result["source_documents"]
                ],
                "question": question,
                "user_id": user_id
            }
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search without generation."""
        try:
            # Perform similarity search
            docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_metadata
                )
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                }
                for doc, score in docs
            ]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def delete_documents(self, filter_metadata: Dict[str, Any]) -> bool:
        """Delete documents from vector store."""
        try:
            self.vector_store.delete(where=filter_metadata)
            self.vector_store.persist()
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

# Global RAG chain instance
prsm_rag_chain = PRSMRAGChain()
```

### 2. Agent-Based Workflows

```python
# prsm/integrations/langchain/agents.py
import asyncio
from typing import List, Dict, Any, Optional, Type
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
import json
import requests

from prsm.integrations.langchain.basic import LangChainPRSMIntegration

class PRSMCalculatorTool(BaseTool):
    """Custom calculator tool for PRSM."""
    
    name = "calculator"
    description = "Useful for mathematical calculations. Input should be a mathematical expression."
    
    def _run(self, query: str) -> str:
        """Execute the calculation."""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in query):
                return "Error: Invalid characters in expression"
            
            result = eval(query, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the calculator."""
        return self._run(query)

class PRSMWeatherTool(BaseTool):
    """Weather information tool for PRSM."""
    
    name = "weather"
    description = "Get current weather information for a city. Input should be a city name."
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    
    def _run(self, city: str) -> str:
        """Get weather information."""
        if not self.api_key:
            return "Weather API key not configured"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            weather_info = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
            
            return json.dumps(weather_info, indent=2)
            
        except Exception as e:
            return f"Error getting weather data: {str(e)}"
    
    async def _arun(self, city: str) -> str:
        """Async version of weather tool."""
        return await asyncio.get_event_loop().run_in_executor(None, self._run, city)

class PRSMKnowledgeBaseTool(BaseTool):
    """Tool to search PRSM's knowledge base."""
    
    name = "knowledge_base"
    description = "Search PRSM's internal knowledge base for information."
    
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Search the knowledge base."""
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found in knowledge base."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Result {i+1}: {doc.page_content[:200]}...")
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of knowledge base search."""
        return await asyncio.get_event_loop().run_in_executor(None, self._run, query)

class PRSMAgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking agent actions."""
    
    def __init__(self):
        self.actions = []
        self.tool_usage = {}
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Track agent actions."""
        self.actions.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        })
        
        # Count tool usage
        tool_name = action.tool
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        """Track agent completion."""
        logger.info(f"Agent completed with {len(self.actions)} actions")

class PRSMAgentFramework:
    """Advanced agent framework for PRSM."""
    
    def __init__(self, vector_store=None):
        self.base_integration = LangChainPRSMIntegration()
        self.callback_handler = PRSMAgentCallbackHandler()
        
        # Initialize tools
        self.tools = self._initialize_tools(vector_store)
        
        # Initialize agents
        self.general_agent = self._create_general_agent()
        self.python_agent = self._create_python_agent()
    
    def _initialize_tools(self, vector_store=None) -> List[Tool]:
        """Initialize available tools."""
        tools = [
            PRSMCalculatorTool(),
            PRSMWeatherTool(),
            DuckDuckGoSearchRun()
        ]
        
        if vector_store:
            tools.append(PRSMKnowledgeBaseTool(vector_store))
        
        return tools
    
    def _create_general_agent(self):
        """Create a general-purpose agent."""
        return initialize_agent(
            tools=self.tools,
            llm=self.base_integration.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            callbacks=[self.callback_handler],
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def _create_python_agent(self):
        """Create a Python code execution agent."""
        return create_python_agent(
            llm=self.base_integration.llm,
            tool=PythonREPLTool(),
            verbose=True,
            callbacks=[self.callback_handler]
        )
    
    async def execute_general_task(self, task: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a general task using the agent."""
        try:
            # Reset callback handler
            self.callback_handler.actions = []
            self.callback_handler.tool_usage = {}
            
            # Execute the task
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.general_agent.run,
                task
            )
            
            return {
                "result": result,
                "actions_taken": self.callback_handler.actions,
                "tool_usage": self.callback_handler.tool_usage,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Agent task execution failed: {e}")
            return {
                "result": f"Task execution failed: {str(e)}",
                "actions_taken": self.callback_handler.actions,
                "tool_usage": self.callback_handler.tool_usage,
                "user_id": user_id
            }
    
    async def execute_python_task(self, code_task: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a Python coding task."""
        try:
            # Reset callback handler
            self.callback_handler.actions = []
            self.callback_handler.tool_usage = {}
            
            # Execute the Python task
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.python_agent.run,
                code_task
            )
            
            return {
                "result": result,
                "actions_taken": self.callback_handler.actions,
                "tool_usage": self.callback_handler.tool_usage,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Python agent task execution failed: {e}")
            return {
                "result": f"Python task execution failed: {str(e)}",
                "actions_taken": self.callback_handler.actions,
                "tool_usage": self.callback_handler.tool_usage,
                "user_id": user_id
            }
    
    def add_custom_tool(self, tool: BaseTool):
        """Add a custom tool to the agent."""
        self.tools.append(tool)
        # Recreate the agent with the new tool
        self.general_agent = self._create_general_agent()
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]

# Global agent framework
prsm_agent_framework = PRSMAgentFramework()
```

## 🔧 Custom Chain Implementations

### 1. Multi-Step Reasoning Chain

```python
# prsm/integrations/langchain/reasoning_chain.py
from typing import Dict, Any, List, Optional
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from pydantic import BaseModel

class ReasoningStep(BaseModel):
    """Single reasoning step."""
    step_number: int
    question: str
    answer: str
    confidence: float

class PRSMReasoningChain(Chain):
    """Multi-step reasoning chain for complex problems."""
    
    llm: BaseLanguageModel
    decomposition_prompt: PromptTemplate
    reasoning_prompt: PromptTemplate
    synthesis_prompt: PromptTemplate
    max_steps: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return ["question"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["final_answer", "reasoning_steps", "confidence_score"]
    
    def __init__(self, llm: BaseLanguageModel, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
        # Decomposition prompt
        self.decomposition_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Break down the following complex question into smaller, manageable sub-questions:

Question: {question}

Sub-questions (list 3-5 specific sub-questions):
1."""
        )
        
        # Reasoning prompt
        self.reasoning_prompt = PromptTemplate(
            input_variables=["sub_question", "context"],
            template="""Answer the following sub-question based on the context provided:

Context: {context}
Sub-question: {sub_question}

Provide a clear, concise answer and rate your confidence (0-1):
Answer:"""
        )
        
        # Synthesis prompt
        self.synthesis_prompt = PromptTemplate(
            input_variables=["original_question", "reasoning_steps"],
            template="""Based on the following reasoning steps, provide a comprehensive final answer:

Original Question: {original_question}

Reasoning Steps:
{reasoning_steps}

Final Answer:"""
        )
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Execute the reasoning chain."""
        question = inputs["question"]
        
        # Step 1: Decompose the question
        decomposition_result = self.llm.predict(
            self.decomposition_prompt.format(question=question)
        )
        
        sub_questions = self._parse_sub_questions(decomposition_result)
        
        # Step 2: Answer each sub-question
        reasoning_steps = []
        context = ""
        
        for i, sub_question in enumerate(sub_questions[:self.max_steps]):
            step_result = self.llm.predict(
                self.reasoning_prompt.format(
                    sub_question=sub_question,
                    context=context
                )
            )
            
            answer, confidence = self._parse_step_result(step_result)
            
            step = ReasoningStep(
                step_number=i + 1,
                question=sub_question,
                answer=answer,
                confidence=confidence
            )
            
            reasoning_steps.append(step)
            context += f"Q{i+1}: {sub_question}\nA{i+1}: {answer}\n\n"
        
        # Step 3: Synthesize final answer
        reasoning_text = "\n".join([
            f"Step {step.step_number}: {step.question}\nAnswer: {step.answer} (Confidence: {step.confidence})"
            for step in reasoning_steps
        ])
        
        final_answer = self.llm.predict(
            self.synthesis_prompt.format(
                original_question=question,
                reasoning_steps=reasoning_text
            )
        )
        
        # Calculate overall confidence
        confidence_score = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        return {
            "final_answer": final_answer,
            "reasoning_steps": [step.dict() for step in reasoning_steps],
            "confidence_score": confidence_score
        }
    
    def _parse_sub_questions(self, decomposition_result: str) -> List[str]:
        """Parse sub-questions from decomposition result."""
        lines = decomposition_result.strip().split('\n')
        sub_questions = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                cleaned = line.lstrip('0123456789.-) ').strip()
                if cleaned:
                    sub_questions.append(cleaned)
        
        return sub_questions
    
    def _parse_step_result(self, step_result: str) -> tuple[str, float]:
        """Parse answer and confidence from step result."""
        lines = step_result.strip().split('\n')
        answer = ""
        confidence = 0.5  # Default confidence
        
        for line in lines:
            if line.strip():
                if 'confidence' in line.lower():
                    # Try to extract confidence score
                    try:
                        import re
                        numbers = re.findall(r'0?\.\d+|\d+', line)
                        if numbers:
                            confidence = float(numbers[0])
                            if confidence > 1:
                                confidence = confidence / 100  # Convert percentage
                    except:
                        pass
                else:
                    answer += line + " "
        
        return answer.strip(), min(max(confidence, 0.0), 1.0)

# Usage example
def create_reasoning_chain(llm):
    """Create a reasoning chain."""
    return PRSMReasoningChain(llm=llm)
```

### 2. Conversational Memory Chain

```python
# prsm/integrations/langchain/memory_chain.py
import json
from typing import Dict, Any, List, Optional
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class PRSMConversationalMemory:
    """Advanced conversational memory for PRSM."""
    
    def __init__(self, llm, max_token_limit: int = 4000):
        self.llm = llm
        self.max_token_limit = max_token_limit
        
        # User-specific memories
        self.user_memories: Dict[str, ConversationSummaryBufferMemory] = {}
        
        # Global conversation chain template
        self.conversation_template = ChatPromptTemplate.from_messages([
            ("system", """You are PRSM, an advanced AI assistant. You maintain context across conversations and learn from user interactions.

Current conversation summary: {history}

User profile:
- Preferences: {user_preferences}
- Previous topics: {previous_topics}
- Communication style: {communication_style}

Respond naturally and helpfully, maintaining context from previous interactions."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    def get_user_memory(self, user_id: str) -> ConversationSummaryBufferMemory:
        """Get or create memory for a specific user."""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
        return self.user_memories[user_id]
    
    async def process_with_memory(
        self,
        user_id: str,
        message: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process message with user-specific memory."""
        memory = self.get_user_memory(user_id)
        
        # Create conversation chain for this user
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            prompt=self.conversation_template,
            verbose=True
        )
        
        # Add user profile information
        profile = user_profile or {}
        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            chain.predict,
            input=message,
            user_preferences=profile.get('preferences', 'Not specified'),
            previous_topics=', '.join(profile.get('topics', [])),
            communication_style=profile.get('style', 'Professional')
        )
        
        return {
            "response": response,
            "memory_summary": memory.predict_new_summary(
                memory.chat_memory.messages,
                ""
            ),
            "message_count": len(memory.chat_memory.messages),
            "user_id": user_id
        }
    
    def get_conversation_summary(self, user_id: str) -> str:
        """Get conversation summary for a user."""
        if user_id not in self.user_memories:
            return "No conversation history"
        
        memory = self.user_memories[user_id]
        return memory.predict_new_summary(memory.chat_memory.messages, "")
    
    def export_conversation(self, user_id: str) -> Dict[str, Any]:
        """Export conversation history for a user."""
        if user_id not in self.user_memories:
            return {"messages": [], "summary": ""}
        
        memory = self.user_memories[user_id]
        
        messages = []
        for message in memory.chat_memory.messages:
            messages.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": getattr(message, 'timestamp', None)
            })
        
        return {
            "messages": messages,
            "summary": self.get_conversation_summary(user_id),
            "message_count": len(messages)
        }
    
    def clear_user_memory(self, user_id: str):
        """Clear memory for a specific user."""
        if user_id in self.user_memories:
            self.user_memories[user_id].clear()
    
    def get_all_user_summaries(self) -> Dict[str, str]:
        """Get summaries for all users."""
        summaries = {}
        for user_id in self.user_memories:
            summaries[user_id] = self.get_conversation_summary(user_id)
        return summaries
```

## 📊 Monitoring and Analytics

### LangChain Integration Metrics

```python
# prsm/integrations/langchain/monitoring.py
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from prometheus_client import Counter, Histogram, Gauge
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Prometheus metrics
langchain_requests_total = Counter(
    'prsm_langchain_requests_total',
    'Total LangChain requests',
    ['chain_type', 'user_id', 'status']
)

langchain_duration_seconds = Histogram(
    'prsm_langchain_duration_seconds',
    'LangChain request duration',
    ['chain_type']
)

langchain_tokens_used = Counter(
    'prsm_langchain_tokens_used_total',
    'Total tokens used by LangChain',
    ['model', 'token_type']
)

langchain_agent_actions = Counter(
    'prsm_langchain_agent_actions_total',
    'Total agent actions',
    ['tool_name', 'status']
)

@dataclass
class LangChainMetrics:
    """LangChain performance metrics."""
    timestamp: datetime
    chain_type: str
    user_id: str
    duration: float
    tokens_used: int
    model: str
    status: str
    error_message: Optional[str] = None
    agent_actions: Optional[List[Dict[str, Any]]] = None

class PRSMLangChainMonitor(BaseCallbackHandler):
    """Comprehensive monitoring for LangChain integration."""
    
    def __init__(self):
        self.start_time = None
        self.chain_type = "unknown"
        self.user_id = "anonymous"
        self.tokens_used = 0
        self.agent_actions = []
        self.status = "success"
        self.error_message = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when chain starts."""
        self.start_time = time.time()
        self.chain_type = serialized.get("name", "unknown")
        self.user_id = inputs.get("user_id", "anonymous")
        
        logger.info(f"LangChain {self.chain_type} started for user {self.user_id}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when chain ends successfully."""
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record metrics
            langchain_requests_total.labels(
                chain_type=self.chain_type,
                user_id=self.user_id,
                status="success"
            ).inc()
            
            langchain_duration_seconds.labels(
                chain_type=self.chain_type
            ).observe(duration)
            
            # Log completion
            logger.info(
                f"LangChain {self.chain_type} completed in {duration:.2f}s "
                f"using {self.tokens_used} tokens"
            )
            
            # Store metrics
            self._store_metrics("success", duration)
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Called when chain errors."""
        if self.start_time:
            duration = time.time() - self.start_time
            
            self.status = "error"
            self.error_message = str(error)
            
            # Record error metrics
            langchain_requests_total.labels(
                chain_type=self.chain_type,
                user_id=self.user_id,
                status="error"
            ).inc()
            
            logger.error(f"LangChain {self.chain_type} failed: {error}")
            
            # Store metrics
            self._store_metrics("error", duration)
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM completes."""
        if hasattr(response, 'llm_output') and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            
            # Record token usage
            for token_type, count in token_usage.items():
                if isinstance(count, int):
                    self.tokens_used += count
                    langchain_tokens_used.labels(
                        model=response.llm_output.get('model_name', 'unknown'),
                        token_type=token_type
                    ).inc(count)
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        action_data = {
            "tool": action.tool,
            "input": action.tool_input,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.agent_actions.append(action_data)
        
        # Record agent action metric
        langchain_agent_actions.labels(
            tool_name=action.tool,
            status="executed"
        ).inc()
        
        logger.debug(f"Agent used tool: {action.tool}")
    
    def _store_metrics(self, status: str, duration: float):
        """Store detailed metrics."""
        metrics = LangChainMetrics(
            timestamp=datetime.utcnow(),
            chain_type=self.chain_type,
            user_id=self.user_id,
            duration=duration,
            tokens_used=self.tokens_used,
            model="gpt-3.5-turbo",  # Default, should be dynamic
            status=status,
            error_message=self.error_message,
            agent_actions=self.agent_actions
        )
        
        # Store in database or cache for analysis
        # This would integrate with your existing metrics storage
        logger.info(f"Stored LangChain metrics: {asdict(metrics)}")

class LangChainAnalytics:
    """Analytics for LangChain integration."""
    
    def __init__(self):
        self.metrics_store = []  # In production, use database
    
    def get_usage_summary(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage summary for date range."""
        # Filter metrics by date range
        filtered_metrics = [
            m for m in self.metrics_store
            if start_date <= m.timestamp <= end_date
        ]
        
        if not filtered_metrics:
            return {"message": "No data for specified range"}
        
        # Calculate statistics
        total_requests = len(filtered_metrics)
        successful_requests = len([m for m in filtered_metrics if m.status == "success"])
        total_tokens = sum(m.tokens_used for m in filtered_metrics)
        avg_duration = sum(m.duration for m in filtered_metrics) / total_requests
        
        # Chain type distribution
        chain_types = {}
        for metric in filtered_metrics:
            chain_types[metric.chain_type] = chain_types.get(metric.chain_type, 0) + 1
        
        # User activity
        user_activity = {}
        for metric in filtered_metrics:
            user_activity[metric.user_id] = user_activity.get(metric.user_id, 0) + 1
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests * 100,
                "total_tokens_used": total_tokens,
                "average_duration": avg_duration,
                "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0
            },
            "chain_type_distribution": chain_types,
            "top_users": sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        recent_metrics = self.metrics_store[-1000:]  # Last 1000 requests
        
        if not recent_metrics:
            return {"message": "Insufficient data"}
        
        # Performance analysis
        durations = [m.duration for m in recent_metrics]
        token_usage = [m.tokens_used for m in recent_metrics]
        
        insights = {
            "performance": {
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "p95_duration": sorted(durations)[int(len(durations) * 0.95)]
            },
            "token_efficiency": {
                "avg_tokens": sum(token_usage) / len(token_usage),
                "max_tokens": max(token_usage),
                "total_cost_estimate": sum(token_usage) * 0.002 / 1000  # Rough estimate
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if insights["performance"]["avg_duration"] > 10:
            insights["recommendations"].append(
                "Consider optimizing prompts or using faster models for better response times"
            )
        
        if insights["token_efficiency"]["avg_tokens"] > 1500:
            insights["recommendations"].append(
                "High token usage detected. Consider prompt optimization or context trimming"
            )
        
        return insights

# Global monitoring instances
langchain_monitor = PRSMLangChainMonitor()
langchain_analytics = LangChainAnalytics()
```

## 📋 Production Integration

### FastAPI Integration

```python
# prsm/api/langchain_endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from prsm.integrations.langchain.basic import langchain_integration
from prsm.integrations.langchain.rag_chain import prsm_rag_chain
from prsm.integrations.langchain.agents import prsm_agent_framework
from prsm.models.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1/langchain", tags=["LangChain"])

class RAGQuery(BaseModel):
    question: str
    user_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class AgentTask(BaseModel):
    task: str
    agent_type: str = "general"  # or "python"
    user_id: Optional[str] = None

class DocumentAdd(BaseModel):
    documents: List[Dict[str, Any]]

@router.post("/query", response_model=QueryResponse)
async def langchain_query(request: QueryRequest):
    """Process query using LangChain integration."""
    try:
        response = await langchain_integration.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/query")
async def rag_query(request: RAGQuery):
    """Query using Retrieval-Augmented Generation."""
    try:
        result = await prsm_rag_chain.query_with_context(
            question=request.question,
            user_id=request.user_id,
            filter_metadata=request.filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/documents")
async def add_documents(request: DocumentAdd):
    """Add documents to the RAG knowledge base."""
    try:
        doc_ids = await prsm_rag_chain.add_documents(request.documents)
        return {"document_ids": doc_ids, "count": len(doc_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent/execute")
async def execute_agent_task(request: AgentTask):
    """Execute task using LangChain agents."""
    try:
        if request.agent_type == "python":
            result = await prsm_agent_framework.execute_python_task(
                request.task,
                request.user_id
            )
        else:
            result = await prsm_agent_framework.execute_general_task(
                request.task,
                request.user_id
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/tools")
async def get_available_tools():
    """Get list of available agent tools."""
    tools = prsm_agent_framework.get_available_tools()
    return {"tools": tools}

@router.get("/memory/summary/{user_id}")
async def get_memory_summary(user_id: str):
    """Get conversation memory summary for user."""
    summary = langchain_integration.get_memory_summary()
    return {"user_id": user_id, "memory_summary": summary}

@router.delete("/memory/{user_id}")
async def clear_user_memory(user_id: str):
    """Clear conversation memory for user."""
    langchain_integration.clear_memory(user_id)
    return {"message": f"Memory cleared for user {user_id}"}
```

---

**Need help with LangChain integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).