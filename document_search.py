"""
Simplified RAG Server for Testing
A lightweight version of the RAG MCP server for easier testing and development

This provides the same functionality as the full MCP server but with a simpler interface
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from document_processor import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call"""
    name: str
    arguments: Dict[str, Any]

@dataclass  
class ToolResult:
    """Represents a tool result"""
    success: bool
    data: Any
    error: Optional[str] = None

class SimpleRAGServer:
    """Simplified RAG server for testing"""
    
    def __init__(self):
        self.rag_system = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the RAG server"""
        try:
            self.rag_system = RAGSystem()
            self.rag_system.initialize()
            self.initialized = True
            logger.info("Simple RAG server initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG server: {e}")
            return False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        return [
            {
                "name": "search_documents",
                "description": "Search shopfloor documents using semantic similarity",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "k": {"type": "integer", "default": 5},
                    "include_metadata": {"type": "boolean", "default": True}
                }
            },
            {
                "name": "get_context", 
                "description": "Get formatted context for LLM augmentation",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "k": {"type": "integer", "default": 3}
                }
            },
            {
                "name": "get_system_stats",
                "description": "Get RAG system statistics",
                "parameters": {}
            },
            {
                "name": "initialize_rag",
                "description": "Initialize or reinitialize the RAG system",
                "parameters": {
                    "force_rebuild": {"type": "boolean", "default": False}
                }
            },
            {
                "name": "search_by_type",
                "description": "Search documents filtered by document type",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "doc_type": {"type": "string", "required": True},
                    "k": {"type": "integer", "default": 5}
                }
            }
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool"""
        try:
            if not self.initialized and tool_name != "initialize_rag":
                await self.initialize()
            
            if tool_name == "search_documents":
                return await self._search_documents(arguments)
            elif tool_name == "get_context":
                return await self._get_context(arguments)
            elif tool_name == "get_system_stats":
                return await self._get_system_stats(arguments)
            elif tool_name == "initialize_rag":
                return await self._initialize_rag(arguments)
            elif tool_name == "search_by_type":
                return await self._search_by_type(arguments)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}"
                )
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _search_documents(self, arguments: Dict[str, Any]) -> ToolResult:
        """Search documents tool"""
        query = arguments["query"]
        k = arguments.get("k", 5)
        include_metadata = arguments.get("include_metadata", True)
        
        result = self.rag_system.query(query, k=k, include_metadata=include_metadata)
        
        return ToolResult(
            success=True,
            data={
                "query": result["query"],
                "num_results": result["num_results"],
                "documents": result["retrieved_documents"]
            }
        )
    
    async def _get_context(self, arguments: Dict[str, Any]) -> ToolResult:
        """Get context tool"""
        query = arguments["query"]
        k = arguments.get("k", 3)
        
        result = self.rag_system.query(query, k=k, include_metadata=False)
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "context": result["context"],
                "num_documents_used": result["num_results"]
            }
        )
    
    async def _get_system_stats(self, arguments: Dict[str, Any]) -> ToolResult:
        """Get system statistics"""
        if not self.rag_system:
            stats = {
                "status": "not_initialized",
                "total_documents": 0,
                "vector_store_built": False
            }
        else:
            stats = self.rag_system.get_statistics()
            stats["status"] = "initialized"
        
        return ToolResult(success=True, data=stats)
    
    async def _initialize_rag(self, arguments: Dict[str, Any]) -> ToolResult:
        """Initialize RAG system"""
        force_rebuild = arguments.get("force_rebuild", False)
        
        try:
            self.rag_system = RAGSystem()
            self.rag_system.initialize(force_rebuild=force_rebuild)
            self.initialized = True
            
            stats = self.rag_system.get_statistics()
            
            return ToolResult(
                success=True,
                data={
                    "status": "success",
                    "message": "RAG system initialized successfully",
                    "statistics": stats
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to initialize RAG system: {str(e)}"
            )
    
    async def _search_by_type(self, arguments: Dict[str, Any]) -> ToolResult:
        """Search documents by type"""
        query = arguments["query"]
        doc_type = arguments["doc_type"]
        k = arguments.get("k", 5)
        
        # Get all results first
        all_results = self.rag_system.query(query, k=k*3, include_metadata=True)
        
        # Filter by document type
        filtered_docs = []
        for doc in all_results["retrieved_documents"]:
            if doc["doc_type"] == doc_type:
                filtered_docs.append(doc)
                if len(filtered_docs) >= k:
                    break
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "doc_type_filter": doc_type,
                "num_results": len(filtered_docs),
                "documents": filtered_docs
            }
        )

# Simplified RAG Agent that uses the simple server
class SimpleRAGAgent:
    """Simplified RAG agent using the simple server"""
    
    def __init__(self):
        self.server = SimpleRAGServer()
        self.initialized = False
        self.stats = {
            'queries_processed': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize the agent"""
        success = await self.server.initialize()
        self.initialized = success
        return success
    
    async def search_documents(self, query: str, k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """Search for documents"""
        import time
        start_time = time.time()
        
        try:
            result = await self.server.call_tool("search_documents", {
                "query": query,
                "k": k,
                "include_metadata": include_metadata
            })
            
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            if result.success:
                data = result.data
                return {
                    'success': True,
                    'query': data["query"],
                    'num_results': data["num_results"],
                    'retrieved_documents': data["documents"],
                    'execution_time': execution_time
                }
            else:
                return {
                    'success': False,
                    'error': result.error,
                    'query': query,
                    'execution_time': execution_time
                }
                
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'execution_time': time.time() - start_time
            }
    
    async def get_context(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Get formatted context"""
        import time
        start_time = time.time()
        
        try:
            result = await self.server.call_tool("get_context", {
                "query": query,
                "k": k
            })
            
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            if result.success:
                data = result.data
                return {
                    'success': True,
                    'query': data["query"],
                    'context': data["context"],
                    'num_documents_used': data["num_documents_used"],
                    'execution_time': execution_time
                }
            else:
                return {
                    'success': False,
                    'error': result.error,
                    'query': query,
                    'execution_time': execution_time
                }
                
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'execution_time': time.time() - start_time
            }
    
    def _update_stats(self, execution_time: float):
        """Update statistics"""
        self.stats['queries_processed'] += 1
        self.stats['total_execution_time'] += execution_time
        self.stats['average_execution_time'] = (
            self.stats['total_execution_time'] / self.stats['queries_processed']
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'active': self.initialized,
            'type': 'SimpleRAGAgent',
            'stats': self.stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.initialized = False

# Test the simple server
async def test_simple_server():
    """Test the simple RAG server"""
    server = SimpleRAGServer()
    
    try:
        print("🧪 Testing Simple RAG Server")
        print("=" * 30)
        
        # Initialize
        print("Initializing server...")
        success = await server.initialize()
        print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if not success:
            return
        
        # List tools
        tools = server.list_tools()
        print(f"Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test search
        print("\nTesting document search...")
        result = await server.call_tool("search_documents", {
            "query": "high vibration machines",
            "k": 3
        })
        
        if result.success:
            print(f"✅ Search successful: {result.data['num_results']} results")
        else:
            print(f"❌ Search failed: {result.error}")
        
        # Test context
        print("\nTesting context generation...")
        result = await server.call_tool("get_context", {
            "query": "maintenance procedures",
            "k": 2
        })
        
        if result.success:
            print(f"✅ Context generated: {len(result.data['context'])} characters")
        else:
            print(f"❌ Context failed: {result.error}")
        
        # Test stats
        print("\nTesting system stats...")
        result = await server.call_tool("get_system_stats", {})
        
        if result.success:
            stats = result.data
            print(f"✅ Stats retrieved:")
            print(f"  - Total documents: {stats.get('total_documents', 0)}")
            print(f"  - Status: {stats.get('status', 'unknown')}")
        else:
            print(f"❌ Stats failed: {result.error}")
        
        print("\n🎉 Simple server test completed!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

async def test_simple_agent():
    """Test the simple RAG agent"""
    agent = SimpleRAGAgent()
    
    try:
        print("\n🤖 Testing Simple RAG Agent")
        print("=" * 30)
        
        # Initialize
        success = await agent.initialize()
        print(f"Agent initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if not success:
            return
        
        # Test search
        result = await agent.search_documents("sensor monitoring", k=2)
        print(f"Document search: {'✅ Success' if result['success'] else '❌ Failed'}")
        if result['success']:
            print(f"  Found {result['num_results']} documents")
        
        # Test context
        result = await agent.get_context("production batches", k=2)
        print(f"Context generation: {'✅ Success' if result['success'] else '❌ Failed'}")
        if result['success']:
            print(f"  Context length: {len(result.get('context', ''))} chars")
        
        # Show stats
        status = agent.get_status()
        print(f"Agent stats: {status['stats']['queries_processed']} queries processed")
        
        await agent.cleanup()
        print("🎉 Simple agent test completed!")
        
    except Exception as e:
        print(f"❌ Agent test error: {e}")

async def main():
    """Main test entry point"""
    await test_simple_server()
    await test_simple_agent()

if __name__ == "__main__":
    asyncio.run(main())