"""
OpenAI Integration for WFC AI System
Combines RAG-retrieved context with OpenAI's GPT models for intelligent responses

This module:
1. Loads OpenAI API configuration from environment
2. Uses retrieved context to generate responses
3. Provides different response modes (simple, detailed, technical)
4. Handles API errors and rate limiting
5. Supports OpenAI Function Calling for tool selection
6. Integrates with ACP for intelligent decision-making
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, using system environment variables")

# OpenAI import with error handling
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration"""
    api_key: str
    model: str = "gpt-4"
    max_tokens: int = 1500
    temperature: float = 0.7
    timeout: int = 30

@dataclass
class LLMResponse:
    """Response from LLM"""
    success: bool
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    function_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_results: Optional[Dict[str, Any]] = None

@dataclass 
class FunctionCallResult:
    """Result from a function call"""
    name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0

class OpenAIIntegration:
    """OpenAI integration for generating responses with RAG context"""
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or self._load_config()
        self.client = None
        self.initialized = False
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost_estimate': 0.0
        }
        
        if OPENAI_AVAILABLE and self.config.api_key:
            self._initialize_client()
    
    def _load_config(self) -> OpenAIConfig:
        """Load configuration from environment variables"""
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        return OpenAIConfig(
            api_key=api_key,
            model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1500')),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            timeout=int(os.getenv('OPENAI_TIMEOUT', '30'))
        )
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if not self.config.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            self.client = OpenAI(api_key=self.config.api_key)
            self.initialized = True
            logger.info("OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if OpenAI integration is available"""
        return OPENAI_AVAILABLE and self.initialized and bool(self.config.api_key)
    
    async def generate_response(
        self, 
        query: str, 
        context: str = "", 
        response_mode: str = "balanced",
        system_prompt: Optional[str] = None,
        use_functions: bool = False,
        available_functions: Optional[List[Dict[str, Any]]] = None,
        function_executor: Optional[Callable] = None
    ) -> LLMResponse:
        """Generate response using OpenAI with RAG context"""
        
        if not self.is_available():
            return LLMResponse(
                success=False,
                content="OpenAI integration not available. Please check API key and installation.",
                error="OpenAI not configured"
            )
        
        start_time = time.time()
        self.stats['requests_made'] += 1
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, context, response_mode, system_prompt)
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            # Extract response
            content = response.choices[0].message.content
            usage = response.usage
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            
            if usage:
                self.stats['total_tokens_used'] += usage.total_tokens
                # Rough cost estimate (GPT-4 pricing as of 2024)
                cost_estimate = (usage.prompt_tokens * 0.00003) + (usage.completion_tokens * 0.00006)
                self.stats['total_cost_estimate'] += cost_estimate
            
            return LLMResponse(
                success=True,
                content=content,
                usage=usage.model_dump() if usage else None,
                model=self.config.model,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"OpenAI API error: {e}")
            
            return LLMResponse(
                success=False,
                content=f"Error generating response: {str(e)}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_response_with_functions(
        self,
        query: str,
        context: str = "",
        response_mode: str = "balanced",
        available_functions: Optional[List[Dict[str, Any]]] = None,
        function_executor: Optional[Callable] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using OpenAI Function Calling"""
        
        if not self.is_available():
            return LLMResponse(
                success=False,
                content="OpenAI integration not available. Please check API key and installation.",
                error="OpenAI not configured"
            )
        
        start_time = time.time()
        self.stats['requests_made'] += 1
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, context, response_mode, system_prompt)
            
            # Prepare function call parameters
            call_params = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "timeout": self.config.timeout
            }
            
            # Add functions if provided
            if available_functions:
                call_params["tools"] = available_functions
                call_params["tool_choice"] = "auto"
            
            # Make initial API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **call_params
            )
            
            # Check if function calls were made
            message = response.choices[0].message
            function_calls = []
            tool_call_results = {}
            
            if message.tool_calls and function_executor:
                # Execute function calls
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🔧 Executing function: {func_name} with args: {func_args}")
                    
                    func_start = time.time()
                    try:
                        result = await function_executor(func_name, func_args)
                        execution_time = time.time() - func_start
                        
                        function_calls.append({
                            "name": func_name,
                            "arguments": func_args,
                            "result": result,
                            "success": True,
                            "execution_time": execution_time
                        })
                        tool_call_results[func_name] = result
                        
                    except Exception as e:
                        execution_time = time.time() - func_start
                        logger.error(f"Function execution error for {func_name}: {e}")
                        
                        function_calls.append({
                            "name": func_name,
                            "arguments": func_args,
                            "error": str(e),
                            "success": False,
                            "execution_time": execution_time
                        })
                        tool_call_results[func_name] = {"error": str(e)}
                
                # Generate final response with function results
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments
                            }
                        }
                        for call in message.tool_calls
                    ]
                })
                
                # Add function results
                for tool_call, func_call in zip(message.tool_calls, function_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(func_call.get("result", func_call.get("error")))
                    })
                
                # Get final synthesis response
                final_response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                content = final_response.choices[0].message.content
                total_usage = {
                    "prompt_tokens": (response.usage.prompt_tokens if response.usage else 0) + 
                                   (final_response.usage.prompt_tokens if final_response.usage else 0),
                    "completion_tokens": (response.usage.completion_tokens if response.usage else 0) + 
                                       (final_response.usage.completion_tokens if final_response.usage else 0),
                    "total_tokens": (response.usage.total_tokens if response.usage else 0) + 
                                  (final_response.usage.total_tokens if final_response.usage else 0)
                }
                
            else:
                # No function calls, return direct response
                content = message.content
                total_usage = response.usage.model_dump() if response.usage else None
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            
            if total_usage and isinstance(total_usage, dict):
                self.stats['total_tokens_used'] += total_usage.get('total_tokens', 0)
                cost_estimate = (total_usage.get('prompt_tokens', 0) * 0.00003) + \
                              (total_usage.get('completion_tokens', 0) * 0.00006)
                self.stats['total_cost_estimate'] += cost_estimate
            
            return LLMResponse(
                success=True,
                content=content,
                usage=total_usage,
                model=self.config.model,
                execution_time=execution_time,
                function_calls=function_calls,
                tool_call_results=tool_call_results
            )
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"OpenAI Function Calling error: {e}")
            
            return LLMResponse(
                success=False,
                content=f"Error with function calling: {str(e)}",
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _prepare_messages(
        self, 
        query: str, 
        context: str, 
        response_mode: str,
        custom_system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        else:
            system_prompt = self._get_system_prompt(response_mode)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if provided
        if context.strip():
            context_message = f"Here's relevant information from the shopfloor database:\n\n{context}\n\n"
            user_message = f"{context_message}Based on this information, please answer: {query}"
        else:
            user_message = query
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _get_system_prompt(self, response_mode: str) -> str:
        """Get system prompt based on response mode"""
        
        base_prompt = """You are an AI assistant for a manufacturing shopfloor management system. You help with:
- Machine monitoring and maintenance
- Production batch tracking  
- Sensor data analysis
- Work order management
- Operator assignments

You have access to real-time shopfloor data including machines, sensors, operators, work orders, and maintenance logs."""
        
        mode_prompts = {
            "simple": f"{base_prompt}\n\nProvide concise, direct answers. Focus on key facts and actionable information.",
            
            "balanced": f"{base_prompt}\n\nProvide comprehensive but accessible answers. Include relevant details and context while remaining clear.",
            
            "technical": f"{base_prompt}\n\nProvide detailed technical responses. Include specific data, measurements, and technical explanations as appropriate.",
            
            "executive": f"{base_prompt}\n\nProvide executive-level summaries. Focus on business impact, key metrics, and strategic insights."
        }
        
        return mode_prompts.get(response_mode, mode_prompts["balanced"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.stats,
            'average_cost_per_request': (
                self.stats['total_cost_estimate'] / max(1, self.stats['successful_requests'])
            ),
            'success_rate': (
                self.stats['successful_requests'] / max(1, self.stats['requests_made'])
            ),
            'configuration': {
                'model': self.config.model,
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature
            }
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection"""
        if not self.is_available():
            return {
                'success': False,
                'error': 'OpenAI not configured or available'
            }
        
        try:
            response = await self.generate_response(
                "Hello, this is a test message. Please respond with 'Connection successful'.",
                response_mode="simple"
            )
            
            return {
                'success': response.success,
                'message': 'OpenAI connection test completed',
                'response_preview': response.content[:100] + "..." if len(response.content) > 100 else response.content,
                'model': response.model,
                'execution_time': response.execution_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Connection test failed: {str(e)}'
            }

class EnhancedRAGAgent:
    """Enhanced RAG Agent with OpenAI integration"""
    
    def __init__(self, rag_agent, openai_integration: Optional[OpenAIIntegration] = None):
        self.rag_agent = rag_agent
        self.openai = openai_integration or OpenAIIntegration()
        self.stats = {
            'enhanced_queries': 0,
            'rag_only_queries': 0,
            'llm_failures': 0
        }
    
    async def enhanced_search(
        self, 
        query: str, 
        k: int = 5, 
        response_mode: str = "balanced",
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """Enhanced search with RAG + LLM response generation"""
        
        # Get RAG results
        rag_result = await self.rag_agent.search_documents(query, k=k)
        
        if not rag_result['success']:
            return rag_result
        
        # If LLM is not requested or not available, return RAG only
        if not use_llm or not self.openai.is_available():
            self.stats['rag_only_queries'] += 1
            return {
                **rag_result,
                'llm_response': None,
                'mode': 'rag_only'
            }
        
        # Generate enhanced response using LLM
        try:
            context = self._format_context_for_llm(rag_result['retrieved_documents'])
            
            llm_response = await self.openai.generate_response(
                query=query,
                context=context,
                response_mode=response_mode
            )
            
            self.stats['enhanced_queries'] += 1
            
            return {
                **rag_result,
                'llm_response': {
                    'success': llm_response.success,
                    'content': llm_response.content,
                    'model': llm_response.model,
                    'execution_time': llm_response.execution_time,
                    'usage': llm_response.usage
                },
                'mode': 'enhanced',
                'response_mode': response_mode
            }
            
        except Exception as e:
            self.stats['llm_failures'] += 1
            logger.error(f"LLM enhancement failed: {e}")
            
            return {
                **rag_result,
                'llm_response': {
                    'success': False,
                    'error': str(e)
                },
                'mode': 'rag_fallback'
            }
    
    def _format_context_for_llm(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for LLM"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            doc_type = doc.get('doc_type', 'unknown')
            score = doc.get('score', 0)
            
            context_parts.append(f"Document {i} ({doc_type}, relevance: {score:.2f}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced agent statistics"""
        return {
            'enhanced_agent_stats': self.stats,
            'rag_agent_stats': self.rag_agent.get_status() if hasattr(self.rag_agent, 'get_status') else {},
            'openai_stats': self.openai.get_stats()
        }

# Testing function
async def test_openai_integration():
    """Test OpenAI integration"""
    print("🧪 Testing OpenAI Integration")
    print("=" * 30)
    
    integration = OpenAIIntegration()
    
    # Test availability
    available = integration.is_available()
    print(f"OpenAI Available: {'✅ Yes' if available else '❌ No'}")
    
    if not available:
        print("⚠️ OpenAI integration not available. Check your .env file and API key.")
        return
    
    # Test connection
    connection_test = await integration.test_connection()
    print(f"Connection Test: {'✅ Success' if connection_test['success'] else '❌ Failed'}")
    
    if connection_test['success']:
        print(f"Response Preview: {connection_test['response_preview']}")
        print(f"Model: {connection_test['model']}")
        print(f"Execution Time: {connection_test['execution_time']:.3f}s")
    else:
        print(f"Error: {connection_test.get('error')}")
    
    # Test with context
    if connection_test['success']:
        print("\n🔍 Testing with sample context...")
        
        sample_context = """
        Machine M100 - Current Status: Running | Temperature: 75°C | Vibration: 6.2 mm/s
        Sensor S101 - Type: Temperature | Reading: 75.0 | Unit: C | Machine: M100
        Work Order WO203 - Status: Overdue | Machine: M100 | Scheduled: 2025-10-25
        """
        
        response = await integration.generate_response(
            query="What's the status of machine M100?",
            context=sample_context,
            response_mode="balanced"
        )
        
        if response.success:
            print(f"✅ Enhanced Response Generated:")
            print(f"Content: {response.content[:200]}...")
            print(f"Tokens Used: {response.usage.get('total_tokens', 'unknown') if response.usage else 'unknown'}")
        else:
            print(f"❌ Enhanced Response Failed: {response.error}")
    
    # Show stats
    stats = integration.get_stats()
    print(f"\n📊 Usage Stats:")
    print(f"Requests: {stats['requests_made']} (Success: {stats['successful_requests']})")
    print(f"Tokens Used: {stats['total_tokens_used']}")
    print(f"Estimated Cost: ${stats['total_cost_estimate']:.4f}")

if __name__ == "__main__":
    asyncio.run(test_openai_integration())