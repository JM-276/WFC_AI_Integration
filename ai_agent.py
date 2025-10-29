"""
Enhanced LLM-Driven Main Agent
Replaces rule-based classification with AI-powered decision making

This agent:
1. Uses OpenAI for intelligent request routing
2. Coordinates with ACP for tool selection
3. Provides dynamic agent orchestration
4. Learns from decision patterns
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import json

from ai_coordinator import ACPCoordinator, ACPRequest, ACPResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Available agent types"""
    RAG = "rag"
    GRAPH = "graph" 
    HYBRID = "hybrid"
    SYSTEM = "system"
    ACP = "acp"

@dataclass
@dataclass
class ProcessingRequest:
    """Enhanced request with AI context"""
    query: str
    user_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    preferred_response_mode: str = "balanced"
    require_sources: bool = True
    max_execution_time: float = 30.0
    preferred_agent: Optional[str] = None
    max_results: int = 10

@dataclass
class ProcessingResponse:
    """Enhanced response with decision metadata"""
    success: bool
    content: str
    agent_used: str
    execution_time: float
    confidence_score: Optional[float] = None
    decision_reasoning: Optional[str] = None
    sources_used: Optional[List[str]] = None
    function_calls_made: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None

class EnhancedMainAgent:
    """LLM-driven main agent with ACP integration"""
    
    def __init__(self):
        self.acp_coordinator = None
        self.registered_agents = {}
        self.decision_history = []
        self.performance_metrics = {
            'requests_processed': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'acp_usage_rate': 0.0,
            'agent_selection_accuracy': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the enhanced main agent"""
        try:
            logger.info("🧠 Initializing Enhanced Main Agent with ACP...")
            
            # Initialize ACP Coordinator
            self.acp_coordinator = ACPCoordinator()
            success = await self.acp_coordinator.initialize()
            
            if success:
                logger.info("✅ Enhanced Main Agent initialized with ACP support")
                return True
            else:
                logger.error("❌ ACP Coordinator initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced Main Agent initialization error: {e}")
            return False
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent with the main agent and ACP"""
        self.registered_agents[agent_name] = agent_instance
        
        # Also register with ACP coordinator
        if self.acp_coordinator:
            self.acp_coordinator.register_agent(agent_name, agent_instance)
        
        logger.info(f"🤖 Registered agent: {agent_name}")
    
    async def process_request(
        self, 
        request: ProcessingRequest,
        use_acp: bool = True
    ) -> ProcessingResponse:
        """Process request using AI-driven decision making"""
        
        start_time = time.time()
        self.performance_metrics['requests_processed'] += 1
        
        try:
            logger.info(f"🎯 Processing request with Enhanced Main Agent: {request.query}")
            
            if use_acp and self.acp_coordinator:
                # Use ACP for intelligent processing
                return await self._process_with_acp(request, start_time)
            else:
                # Fallback to enhanced direct processing
                return await self._process_with_enhanced_logic(request, start_time)
                
        except Exception as e:
            logger.error(f"❌ Request processing error: {e}")
            return ProcessingResponse(
                success=False,
                content=f"Processing error: {str(e)}",
                agent_used="main",
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _process_with_acp(
        self, 
        request: ProcessingRequest, 
        start_time: float
    ) -> ProcessingResponse:
        """Process request using ACP (AI Control Protocol)"""
        
        logger.info("🧠 Using ACP for AI-driven processing")
        
        # Convert to ACP format
        acp_request = ACPRequest(
            query=request.query,
            user_context=request.user_context,
            conversation_history=request.conversation_history,
            response_format=request.preferred_response_mode
        )
        
        # Process with ACP
        acp_response = await self.acp_coordinator.process_request(acp_request)
        
        # Update metrics
        self.performance_metrics['acp_usage_rate'] = \
            (self.performance_metrics['acp_usage_rate'] * (self.performance_metrics['requests_processed'] - 1) + 1) / \
            self.performance_metrics['requests_processed']
        
        if acp_response.success:
            self.performance_metrics['successful_requests'] += 1
        
        # Convert ACP response to our format
        return ProcessingResponse(
            success=acp_response.success,
            content=acp_response.final_answer,
            agent_used="acp",
            execution_time=acp_response.execution_time,
            confidence_score=acp_response.confidence_score,
            decision_reasoning=self._format_decision_trail(acp_response.decision_trail),
            sources_used=acp_response.tools_used,
            function_calls_made=acp_response.decision_trail,
            error=acp_response.error,
            suggestions=acp_response.suggestions
        )
    
    async def _process_with_enhanced_logic(
        self, 
        request: ProcessingRequest, 
        start_time: float
    ) -> ProcessingResponse:
        """Fallback processing with enhanced logic"""
        
        logger.info("🔄 Using enhanced fallback processing")
        
        try:
            # Simple intelligent routing based on query analysis
            analysis = await self._analyze_query_intent(request.query)
            
            if analysis['agent_type'] == 'rag':
                result = await self._execute_rag_request(request)
            elif analysis['agent_type'] == 'graph':
                result = await self._execute_graph_request(request)
            else:
                result = await self._execute_hybrid_request(request)
            
            if result['success']:
                self.performance_metrics['successful_requests'] += 1
            
            return ProcessingResponse(
                success=result['success'],
                content=result.get('content', ''),
                agent_used=analysis['agent_type'],
                execution_time=time.time() - start_time,
                decision_reasoning=analysis['reasoning'],
                sources_used=result.get('sources', []),
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Enhanced processing error: {e}")
            return ProcessingResponse(
                success=False,
                content=f"Enhanced processing failed: {str(e)}",
                agent_used="main",
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using simple heuristics"""
        
        query_lower = query.lower()
        
        # Graph query indicators
        graph_indicators = [
            'machine', 'sensor', 'operator', 'work order', 'maintenance', 'zone',
            'vibration', 'overdue', 'current', 'list', 'find', 'show', 'get'
        ]
        
        # RAG query indicators  
        rag_indicators = [
            'documentation', 'procedure', 'how to', 'what is', 'explain', 'describe',
            'information about', 'search for', 'tell me about'
        ]
        
        graph_score = sum(1 for indicator in graph_indicators if indicator in query_lower)
        rag_score = sum(1 for indicator in rag_indicators if indicator in query_lower)
        
        if graph_score > rag_score:
            return {
                'agent_type': 'graph',
                'confidence': graph_score / (graph_score + rag_score) if (graph_score + rag_score) > 0 else 0.5,
                'reasoning': f'Query contains {graph_score} graph indicators vs {rag_score} RAG indicators'
            }
        elif rag_score > graph_score:
            return {
                'agent_type': 'rag',
                'confidence': rag_score / (graph_score + rag_score) if (graph_score + rag_score) > 0 else 0.5,
                'reasoning': f'Query contains {rag_score} RAG indicators vs {graph_score} graph indicators'
            }
        else:
            return {
                'agent_type': 'hybrid',
                'confidence': 0.5,
                'reasoning': 'Equal indicators for both agents, using hybrid approach'
            }
    
    async def _execute_rag_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Execute RAG request"""
        if 'rag' not in self.registered_agents:
            return {'success': False, 'error': 'RAG agent not available'}
        
        try:
            agent = self.registered_agents['rag']
            result = await agent.search_documents(request.query, k=5)
            
            return {
                'success': result.get('success', False),
                'content': self._format_rag_response(result),
                'sources': [f"Document {i+1}" for i in range(len(result.get('data', {}).get('retrieved_documents', [])))]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_graph_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Execute graph request"""
        if 'graph' not in self.registered_agents:
            return {'success': False, 'error': 'Graph agent not available'}
        
        try:
            agent = self.registered_agents['graph']
            
            # Try to determine specific graph operation
            query_lower = request.query.lower()
            
            if 'vibration' in query_lower:
                result = await agent.find_high_vibration_machines()
            elif 'overdue' in query_lower:
                result = await agent.get_overdue_work_orders()
            elif 'operator' in query_lower and any(machine in query_lower for machine in ['m100', 'm200', 'm300']):
                # Extract machine ID
                machine_id = next((word for word in request.query.split() if word.upper().startswith('M')), 'M100')
                result = await agent.get_current_operator(machine_id.upper())
            else:
                # Default to high vibration check
                result = await agent.find_high_vibration_machines()
            
            return {
                'success': result.get('success', False),
                'content': self._format_graph_response(result),
                'sources': ['Graph Database']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_hybrid_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Execute hybrid request using both agents"""
        rag_result = await self._execute_rag_request(request)
        graph_result = await self._execute_graph_request(request)
        
        # Combine results
        combined_content = []
        sources = []
        
        if rag_result.get('success'):
            combined_content.append(f"Document Search Results:\n{rag_result['content']}")
            sources.extend(rag_result.get('sources', []))
        
        if graph_result.get('success'):
            combined_content.append(f"Database Query Results:\n{graph_result['content']}")
            sources.extend(graph_result.get('sources', []))
        
        if combined_content:
            return {
                'success': True,
                'content': '\n\n'.join(combined_content),
                'sources': sources
            }
        else:
            return {
                'success': False,
                'error': 'Both RAG and Graph queries failed'
            }
    
    def _format_rag_response(self, result: Dict[str, Any]) -> str:
        """Format RAG response for display"""
        if not result.get('success'):
            return f"RAG search failed: {result.get('error', 'Unknown error')}"
        
        data = result.get('data', {})
        documents = data.get('retrieved_documents', [])
        
        if not documents:
            return "No relevant documents found."
        
        formatted = f"Found {len(documents)} relevant documents:\n\n"
        for i, doc in enumerate(documents[:3], 1):
            content = doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
            score = doc.get('score', 0)
            formatted += f"{i}. [Relevance: {score:.2f}] {content}\n\n"
        
        return formatted
    
    def _format_graph_response(self, result: Dict[str, Any]) -> str:
        """Format graph response for display"""
        if not result.get('success'):
            return f"Graph query failed: {result.get('error', 'Unknown error')}"
        
        data = result.get('data', {})
        
        if isinstance(data, dict) and 'count' in data:
            count = data['count']
            records = data.get('data', [])
            
            if count == 0:
                return "No matching records found in the database."
            
            formatted = f"Found {count} records:\n\n"
            for i, record in enumerate(records[:5], 1):
                formatted += f"{i}. {record}\n"
            
            if count > 5:
                formatted += f"\n... and {count - 5} more records"
            
            return formatted
        
        elif isinstance(data, list):
            if not data:
                return "No results found."
            
            formatted = f"Query Results ({len(data)} items):\n\n"
            for i, item in enumerate(data[:5], 1):
                formatted += f"{i}. {item}\n"
            
            return formatted
        
        else:
            return str(data)
    
    def _format_decision_trail(self, decision_trail: List[Dict[str, Any]]) -> str:
        """Format decision trail for human reading"""
        if not decision_trail:
            return "No decision trail available"
        
        formatted_steps = []
        for step in decision_trail:
            phase = step.get('phase', 'unknown')
            action = step.get('action', step.get('result', ''))
            formatted_steps.append(f"{phase.title()}: {action}")
        
        return " → ".join(formatted_steps)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Update average response time
        if self.performance_metrics['requests_processed'] > 0:
            self.performance_metrics['average_response_time'] = \
                sum(decision.get('execution_time', 0) for decision in self.decision_history[-100:]) / \
                min(len(self.decision_history), 100) if self.decision_history else 0.0
        
        return self.performance_metrics.copy()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {
            'main_agent': {
                'type': 'EnhancedMainAgent',
                'status': 'active',
                'acp_available': self.acp_coordinator is not None,
                'registered_agents': list(self.registered_agents.keys())
            }
        }
        
        for name, agent in self.registered_agents.items():
            try:
                if hasattr(agent, 'get_status'):
                    status[name] = agent.get_status()
                else:
                    status[name] = {'status': 'active', 'type': type(agent).__name__}
            except Exception as e:
                status[name] = {'status': 'error', 'error': str(e)}
        
        return status

# Test function
async def test_enhanced_main_agent():
    """Test the enhanced main agent"""
    agent = EnhancedMainAgent()
    
    if await agent.initialize():
        print("✅ Enhanced Main Agent initialized successfully")
        
        # Test request
        test_request = ProcessingRequest(
            query="What machines have high vibration and need maintenance?",
            preferred_response_mode="comprehensive"
        )
        
        response = await agent.process_request(test_request)
        
        print(f"Success: {response.success}")
        print(f"Agent Used: {response.agent_used}")
        print(f"Content: {response.content[:200]}...")
        print(f"Execution Time: {response.execution_time:.3f}s")
        print(f"Decision Reasoning: {response.decision_reasoning}")
    else:
        print("❌ Enhanced Main Agent initialization failed")

if __name__ == "__main__":
    asyncio.run(test_enhanced_main_agent())