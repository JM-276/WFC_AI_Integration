"""
Complete Integration Script for WFC AI System
Luke's RAG + Multi-Agent Architecture with ACP Enhancement

This script demonstrates the enhanced system:
1. ACP (AI Control Protocol) for intelligent coordination
2. LLM-driven tool selection and decision making
3. OpenAI Function Calling integration
4. Enhanced main agent with AI reasoning
5. MCP server integration (when available)
6. Fallback to direct agent communication
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import time
from pathlib import Path

# Import enhanced components
try:
    from ai_main_agent import EnhancedMainAgent, ProcessingRequest, ProcessingResponse
    ENHANCED_MAIN_AVAILABLE = True
except ImportError:
    ENHANCED_MAIN_AVAILABLE = False
    from ai_main_agent import EnhancedMainAgent

try:
    from ai_coordinator import ACPCoordinator, ACPRequest
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False

# Import existing components with fallbacks
try:
    from document_processor import DocumentProcessor
except ImportError:
    from document_search import SimpleRAGAgent as RAGAgent

from database_agent import GraphAgent
from document_processor import RAGSystem
from ai_brain import OpenAIIntegration, EnhancedRAGAgent

# Import standard main agent components
# AI-driven request processing - no need for request types in enhanced system

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WFCIntegrationSystem:
    """Complete WFC AI Integration System with ACP Enhancement"""
    
    def __init__(self, use_enhanced_mode: bool = True):
        # Core agents
        self.main_agent = None
        self.rag_agent = None
        self.graph_agent = None
        
        # Enhanced components
        self.enhanced_main_agent = None
        self.acp_coordinator = None
        self.openai_integration = None
        self.enhanced_rag_agent = None
        
        # Configuration
        self.use_enhanced_mode = use_enhanced_mode and ENHANCED_MAIN_AVAILABLE and ACP_AVAILABLE
        self.initialized = False
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'enhanced_requests': 0,
            'acp_requests': 0,
            'agent_usage': {
                'rag': 0,
                'graph': 0,
                'hybrid': 0,
                'enhanced': 0,
                'acp': 0
            }
        }
    
    async def initialize(self):
        """Initialize the complete system with enhanced capabilities"""
        try:
            logger.info("🚀 Initializing WFC AI Integration System...")
            
            if self.use_enhanced_mode:
                logger.info("🧠 Initializing Enhanced AI-Driven System...")
                
                # Initialize Enhanced Main Agent with ACP
                self.enhanced_main_agent = EnhancedMainAgent()
                enhanced_success = await self.enhanced_main_agent.initialize()
                
                if enhanced_success:
                    logger.info("✅ Enhanced Main Agent with ACP initialized")
                    self.main_agent = self.enhanced_main_agent  # Use as main agent
                else:
                    logger.warning("⚠️ Enhanced mode failed, falling back to standard mode")
                    self.use_enhanced_mode = False
            
            if not self.use_enhanced_mode:
                # Fallback to enhanced agent anyway since old system removed
                logger.warning("⚠️ Forcing enhanced mode since standard mode removed")
                self.main_agent = self.enhanced_main_agent
                self.use_enhanced_mode = True
            
            # Initialize RAG agent (use simple version for reliability)
            logger.info("🔍 Initializing RAG agent...")
            try:
                from document_search import SimpleRAGAgent
                self.rag_agent = SimpleRAGAgent()
                rag_success = await self.rag_agent.initialize()
                logger.info("✅ Using SimpleRAGAgent for better reliability")
            except ImportError:
                self.rag_agent = RAGAgent()
                rag_success = await self.rag_agent.initialize()
                logger.info("✅ Using standard RAGAgent")
            
            if rag_success:
                self.main_agent.register_agent('rag', self.rag_agent)
                logger.info("✅ RAG agent initialized and registered")
            else:
                logger.warning("⚠️ RAG agent initialization failed")
            
            # Initialize Graph agent
            logger.info("📊 Initializing Graph agent...")
            self.graph_agent = GraphAgent()
            graph_success = await self.graph_agent.initialize()
            
            if graph_success:
                self.main_agent.register_agent('graph', self.graph_agent)
                logger.info("✅ Graph agent initialized and registered")
            else:
                logger.warning("⚠️ Graph agent initialization failed")
            
            # Initialize OpenAI integration
            logger.info("🤖 Initializing OpenAI integration...")
            self.openai_integration = OpenAIIntegration()
            
            if self.openai_integration.is_available():
                # Test connection
                test_result = await self.openai_integration.test_connection()
                if test_result['success']:
                    logger.info("✅ OpenAI integration initialized and tested")
                    
                    # Create enhanced RAG agent
                    if rag_success:
                        self.enhanced_rag_agent = EnhancedRAGAgent(
                            self.rag_agent, 
                            self.openai_integration
                        )
                        self.main_agent.register_agent('enhanced', self.enhanced_rag_agent)
                        logger.info("✅ Enhanced RAG agent created")
                else:
                    logger.warning(f"⚠️ OpenAI connection test failed: {test_result.get('error')}")
            else:
                logger.warning("⚠️ OpenAI integration not available (check API key in .env)")
            
            self.initialized = True
            logger.info("🎉 WFC AI Integration System fully initialized!")
            
            # Show system status
            await self.show_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def process_query(self, query: str, preferred_agent: str = None, max_results: int = 5, use_llm: bool = True, response_mode: str = "balanced") -> Dict[str, Any]:
        """Process a user query through the system"""
        if not self.initialized:
            return {
                'success': False,
                'error': 'System not initialized',
                'query': query
            }
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Process through enhanced AI-driven main agent
            from ai_main_agent import ProcessingRequest
            request = ProcessingRequest(
                query=query,
                preferred_agent=preferred_agent,
                max_results=max_results
            )
            response = await self.main_agent.process_request(request)
            
            # Update statistics
            if response.get('success', False):
                self.stats['successful_requests'] += 1
                agent_used = response.get('agent_used', 'unknown')
                if agent_used in self.stats['agent_usage']:
                    self.stats['agent_usage'][agent_used] += 1
            else:
                self.stats['failed_requests'] += 1
            
            # Add system metadata
            response['system_execution_time'] = time.time() - start_time
            response['system_stats'] = self.get_quick_stats()
            
            return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'system_execution_time': time.time() - start_time
            }
    
    async def process_enhanced_query(self, query: str, use_llm: bool = True, response_mode: str = "balanced", max_results: int = 5) -> Dict[str, Any]:
        """Process query with enhanced AI-driven capabilities"""
        if not self.initialized:
            return {
                'success': False,
                'error': 'System not initialized',
                'query': query
            }
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Use Enhanced Main Agent with ACP if available
            if self.use_enhanced_mode and self.enhanced_main_agent:
                logger.info(f"🧠 Processing with Enhanced AI-Driven System: {query}")
                
                # Create enhanced processing request
                request = ProcessingRequest(
                    query=query,
                    preferred_response_mode=response_mode,
                    require_sources=True,
                    max_execution_time=30.0
                )
                
                # Process with ACP coordination
                response = await self.enhanced_main_agent.process_request(request, use_acp=True)
                
                if response.success:
                    self.stats['successful_requests'] += 1
                    self.stats['enhanced_requests'] += 1
                    if response.agent_used == 'acp':
                        self.stats['acp_requests'] += 1
                    self.stats['agent_usage'][response.agent_used] += 1
                else:
                    self.stats['failed_requests'] += 1
                
                return {
                    'success': response.success,
                    'query': query,
                    'agent_used': response.agent_used,
                    'data': {
                        'content': response.content,
                        'decision_reasoning': response.decision_reasoning,
                        'sources_used': response.sources_used,
                        'function_calls': response.function_calls_made,
                        'confidence_score': response.confidence_score
                    },
                    'response_mode': response_mode,
                    'execution_time': response.execution_time,
                    'suggestions': response.suggestions,
                    'error': response.error
                }
            
            # Fallback to enhanced RAG with OpenAI
            elif use_llm and self.enhanced_rag_agent and self.openai_integration.is_available():
                logger.info(f"🤖 Processing enhanced query with OpenAI: {query}")
                
                result = await self.enhanced_rag_agent.enhanced_search(
                    query=query,
                    k=max_results,
                    response_mode=response_mode,
                    use_llm=True
                )
                
                self.stats['successful_requests'] += 1
                self.stats['agent_usage']['enhanced'] += 1
                
                return {
                    'success': True,
                    'query': query,
                    'agent_used': 'enhanced_rag',
                    'response_mode': response_mode,
                    'data': result,
                    'llm_response': result.get('llm_response'),
                    'execution_time': time.time() - start_time,
                    'system_stats': self.get_quick_stats()
                }
            
            else:
                # Fallback to regular processing
                logger.info(f"📋 Processing regular query (LLM not available or not requested): {query}")
                return await self.process_query(query, max_results=max_results, use_llm=False)
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Error processing enhanced query '{query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'execution_time': time.time() - start_time
            }
    
    async def process_acp_query(
        self, 
        query: str, 
        response_mode: str = "balanced",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process query specifically using ACP (AI Control Protocol)"""
        if not self.use_enhanced_mode or not self.enhanced_main_agent:
            return {
                'success': False,
                'error': 'ACP mode not available. Enhanced system required.',
                'query': query
            }
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        self.stats['acp_requests'] += 1
        
        try:
            logger.info(f"🧠🤖 Processing query with ACP Intelligence: {query}")
            
            # Create ACP request
            request = ProcessingRequest(
                query=query,
                conversation_history=conversation_history,
                preferred_response_mode=response_mode,
                require_sources=True,
                max_execution_time=45.0  # Allow more time for complex AI reasoning
            )
            
            # Force ACP usage
            response = await self.enhanced_main_agent.process_request(request, use_acp=True)
            
            if response.success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            self.stats['agent_usage']['acp'] += 1
            
            return {
                'success': response.success,
                'query': query,
                'agent_used': response.agent_used,
                'data': {
                    'ai_response': response.content,
                    'decision_trail': response.decision_reasoning,
                    'tools_executed': response.sources_used,
                    'function_calls': response.function_calls_made,
                    'ai_confidence': response.confidence_score
                },
                'response_mode': response_mode,
                'execution_time': response.execution_time,
                'ai_suggestions': response.suggestions,
                'error': response.error,
                'processing_method': 'ACP (AI Control Protocol)'
            }
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"ACP processing error for '{query}': {e}")
            return {
                'success': False,
                'error': f"ACP processing failed: {str(e)}",
                'query': query,
                'execution_time': time.time() - start_time,
                'processing_method': 'ACP (AI Control Protocol)'
            }
    
    async def show_system_status(self):
        """Show current system status including enhanced capabilities"""
        print("\n" + "="*60)
        print("🏭 WFC AI INTEGRATION SYSTEM STATUS")
        print("="*60)
        
        # System mode
        if self.use_enhanced_mode:
            print("🧠 Mode: Enhanced AI-Driven System with ACP")
        else:
            print("📋 Mode: Standard Multi-Agent System")
        
        # Main agent status
        if self.enhanced_main_agent:
            print(f"🧠 Enhanced Main Agent: ✅ Active (with AI Control Protocol)")
            
            # Get enhanced performance metrics
            metrics = self.enhanced_main_agent.get_performance_metrics()
            print(f"   AI Requests: {metrics.get('requests_processed', 0)}")
            print(f"   ACP Usage Rate: {metrics.get('acp_usage_rate', 0)*100:.1f}%")
            
            # Get agent status
            agent_status = self.enhanced_main_agent.get_agent_status()
            for name, status in agent_status.items():
                if name != 'main_agent':
                    active_status = "✅ Active" if status.get('status') == 'active' else "❌ Inactive"
                    print(f"🤖 {name.upper()} Agent: {active_status}")
        
        elif self.main_agent:
            main_status = self.main_agent.get_agent_status()
            print(f"📋 Main Agent: {'✅ Active' if main_status['main_agent']['active'] else '❌ Inactive'}")
            print(f"   Requests processed: {main_status['main_agent']['requests_processed']}")
            
            # Secondary agents
            for agent_name, status in main_status['secondary_agents'].items():
                active_status = "✅ Active" if status.get('active', False) else "❌ Inactive"
                print(f"🤖 {agent_name.upper()} Agent: {active_status}")
                if 'stats' in status:
                    stats = status['stats']
                    if 'queries_processed' in stats:
                        print(f"   Queries: {stats['queries_processed']}, Avg time: {stats.get('average_execution_time', 0):.3f}s")
        
        # System statistics
        print(f"\n📊 System Statistics:")
        print(f"   Total requests: {self.stats['total_requests']}")
        print(f"   Success rate: {(self.stats['successful_requests']/max(1, self.stats['total_requests'])*100):.1f}%")
        
        # Agent usage
        if any(self.stats['agent_usage'].values()):
            print(f"   Agent usage:")
            for agent, count in self.stats['agent_usage'].items():
                if count > 0:
                    print(f"     {agent.upper()}: {count} requests")
        
        print("="*60)
    
    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick system statistics including enhanced metrics"""
        base_stats = {
            'total_requests': self.stats['total_requests'],
            'success_rate': self.stats['successful_requests'] / max(1, self.stats['total_requests']),
            'agent_usage': self.stats['agent_usage']
        }
        
        # Add enhanced stats if available
        if self.use_enhanced_mode:
            base_stats.update({
                'enhanced_requests': self.stats['enhanced_requests'],
                'acp_requests': self.stats['acp_requests'],
                'ai_usage_rate': self.stats['enhanced_requests'] / max(1, self.stats['total_requests']),
                'acp_usage_rate': self.stats['acp_requests'] / max(1, self.stats['total_requests']),
                'system_mode': 'Enhanced AI-Driven'
            })
        else:
            base_stats['system_mode'] = 'Standard Multi-Agent'
        
        return base_stats
    
    async def run_demo(self):
        """Run a comprehensive demo of the system"""
        if not self.initialized:
            logger.error("System not initialized. Call initialize() first.")
            return
        
        print("\n" + "🎮 RUNNING WFC AI INTEGRATION DEMO" + "\n" + "="*60)
        
        # Demo queries showcasing different capabilities
        demo_queries = [
            {
                'query': "What machines have high vibration readings above 7.5 mm/s?",
                'expected_agent': 'graph',
                'description': 'Graph query for specific operational data'
            },
            {
                'query': "Describe the maintenance procedures for production equipment",
                'expected_agent': 'rag',
                'description': 'Semantic search for documentation'
            },
            {
                'query': "Show me current operators for machine M100",
                'expected_agent': 'graph',
                'description': 'Structured query for real-time data'
            },
            {
                'query': "Find information about sensor monitoring systems",
                'expected_agent': 'rag',
                'description': 'General information retrieval'
            },
            {
                'query': "Which work orders are overdue and what maintenance is needed?",
                'expected_agent': 'hybrid',
                'description': 'Hybrid query needing both structured and semantic data'
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n🔍 Demo Query {i}/5")
            print(f"Query: {demo['query']}")
            print(f"Expected: {demo['description']}")
            print("-" * 50)
            
            # Process query
            result = await self.process_query(demo['query'])
            
            # Display results
            print(f"✅ Success: {result.get('success', False)}")
            print(f"🤖 Agent used: {result.get('agent_used', 'unknown')}")
            print(f"⏱️ Execution time: {result.get('execution_time', 0):.3f}s")
            
            if result.get('success'):
                print(f"💡 Explanation: {result.get('explanation', 'No explanation provided')}")
                
                # Show sample data
                data = result.get('data', {})
                if 'retrieved_documents' in data:
                    print(f"📄 Found {len(data['retrieved_documents'])} documents")
                elif 'count' in data:
                    print(f"📊 Found {data['count']} records")
                elif isinstance(data, list):
                    print(f"📊 Found {len(data)} results")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Small delay between queries
            await asyncio.sleep(1)
        
        # Final system status
        await self.show_system_status()
    
    async def interactive_mode(self):
        """Run in interactive mode for testing"""
        if not self.initialized:
            logger.error("System not initialized. Call initialize() first.")
            return
        
        print("\n🎯 INTERACTIVE MODE - Enter queries (type 'quit' to exit)")
        print("Available commands:")
        print("  help - Show this help")
        print("  status - Show system status")
        print("  stats - Show detailed statistics")
        print("  rag <query> - Force RAG agent")
        print("  graph <query> - Force graph agent")
        print("  enhanced <query> - Use OpenAI enhanced responses")
        print("  simple <query> - Simple mode response")
        print("  technical <query> - Technical mode response")
        print("  quit - Exit interactive mode")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n🤔 Enter your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif query.lower() == 'help':
                    print("Available commands: help, status, stats, rag <query>, graph <query>, quit")
                    continue
                
                elif query.lower() == 'status':
                    await self.show_system_status()
                    continue
                
                elif query.lower() == 'stats':
                    print(json.dumps(self.get_quick_stats(), indent=2))
                    continue
                
                # Handle different query types
                preferred_agent = None
                use_enhanced = False
                response_mode = "balanced"
                
                if query.startswith('rag '):
                    preferred_agent = 'rag'
                    query = query[4:]
                elif query.startswith('graph '):
                    preferred_agent = 'graph'
                    query = query[6:]
                elif query.startswith('enhanced '):
                    use_enhanced = True
                    query = query[9:]
                elif query.startswith('simple '):
                    use_enhanced = True
                    response_mode = "simple"
                    query = query[7:]
                elif query.startswith('technical '):
                    use_enhanced = True
                    response_mode = "technical"
                    query = query[10:]
                
                # Process query (enhanced or regular)
                print(f"⏳ Processing: {query}")
                if use_enhanced:
                    result = await self.process_enhanced_query(query, use_llm=True, response_mode=response_mode)
                else:
                    result = await self.process_query(query, preferred_agent)
                
                # Display results
                print(f"\n{'='*50}")
                print(f"✅ Success: {result.get('success', False)}")
                print(f"🤖 Agent: {result.get('agent_used', 'unknown')}")
                print(f"⏱️ Time: {result.get('execution_time', 0):.3f}s")
                
                if result.get('success'):
                    print(f"💡 {result.get('explanation', '')}")
                    
                    # Show LLM response if available
                    llm_response = result.get('llm_response')
                    if llm_response and llm_response.get('success'):
                        print(f"\n🤖 AI Response ({result.get('response_mode', 'balanced')} mode):")
                        print(f"{llm_response['content']}")
                        if llm_response.get('usage'):
                            tokens = llm_response['usage'].get('total_tokens', '?')
                            print(f"   (Tokens used: {tokens})")
                    elif llm_response and not llm_response.get('success'):
                        print(f"\n⚠️ AI Response failed: {llm_response.get('error', 'Unknown error')}")
                    
                    # Show data summary
                    data = result.get('data', {})
                    if 'context' in data:
                        print(f"\n📝 Context:\n{data['context'][:300]}...")
                    if 'retrieved_documents' in data:
                        print(f"\n📄 Found {len(data['retrieved_documents'])} documents")
                    elif 'count' in data:
                        print(f"\n📊 Found {data['count']} records")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("🧹 Cleaning up system resources...")
        
        if self.rag_agent:
            await self.rag_agent.cleanup()
        
        if self.graph_agent:
            self.graph_agent.cleanup()
        
        logger.info("✅ Cleanup complete")

async def main():
    """Main entry point"""
    system = WFCIntegrationSystem()
    
    try:
        # Initialize system
        success = await system.initialize()
        if not success:
            print("❌ Failed to initialize system")
            return
        
        # Run demo
        print("\n🎯 Choose mode:")
        print("1. Run demo (automated)")
        print("2. Interactive mode")
        print("3. Just show status")
        
        try:
            choice = input("Enter choice (1-3): ").strip()
        except KeyboardInterrupt:
            choice = "3"
        
        if choice == "1":
            await system.run_demo()
        elif choice == "2":
            await system.interactive_mode()
        else:
            await system.show_system_status()
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())