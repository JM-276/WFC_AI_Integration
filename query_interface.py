"""
Interactive Query Interface for WFC AI Integration System
Allows you to input your own queries and see real-time responses

This gives you full control to test any questions you want!
"""

import asyncio
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ai_system import WFCIntegrationSystem

class InteractiveQueryInterface:
    """Interactive interface for testing queries"""
    
    def __init__(self):
        self.system = None
        self.openai_available = False
    
    async def initialize(self):
        """Initialize the system"""
        # Silent initialization
        
        self.system = WFCIntegrationSystem()
        success = await self.system.initialize()
        
        if not success:
            print("❌ System initialization failed")
            return False
        
        # Check OpenAI availability
        api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_available = api_key and api_key != 'your_openai_api_key_here'
        

        return True
    
    async def show_help(self):
        """Show available commands and examples"""
        print("\n" + "="*60)
        print("🎯 INTERACTIVE QUERY HELP")
        print("="*60)
        
        print("\n📝 COMMAND FORMATS:")
        print("  <query>              - Auto-detect best agent")
        print("  rag <query>          - Force RAG (semantic search)")
        print("  graph <query>        - Force graph database")
        
        if self.openai_available:
            print("  enhanced <query>     - Use OpenAI (balanced mode)")
            print("  simple <query>       - Use OpenAI (simple explanations)")
            print("  technical <query>    - Use OpenAI (technical details)")
            print("  executive <query>    - Use OpenAI (executive summary)")
        
        print("\n🔧 SYSTEM COMMANDS:")
        print("  help                 - Show this help")
        print("  status               - Show system status")
        print("  stats                - Show usage statistics")
        print("  examples             - Show example queries")
        print("  quit                 - Exit")
        
        print("\n💡 EXAMPLE QUERIES:")
        print("Basic queries:")
        print("  What machines have high vibration?")
        print("  Show current operators")
        print("  Find overdue work orders")
        print("  List sensors in zone Z1")
        
        print("\nRAG queries:")
        print("  Find information about maintenance procedures")
        print("  What documentation exists for sensor monitoring?")
        print("  Search for production batch information")
        
        if self.openai_available:
            print("\nEnhanced queries (with OpenAI):")
            print("  enhanced What should I do about machine M100?")
            print("  simple Explain high vibration in simple terms")
            print("  technical Analyze the sensor data patterns")
            print("  executive Summarize facility operational status")
        
        print("="*60)
    
    async def show_examples(self):
        """Show detailed examples with expected outputs"""
        print("\n" + "="*60)
        print("📚 DETAILED EXAMPLES")
        print("="*60)
        
        examples = [
            {
                "category": "🔍 Graph Database Queries",
                "queries": [
                    "What machines have high vibration above 7.5 mm/s?",
                    "Show overdue work orders",
                    "Who is operating machine M100?",
                    "List all sensors in zone Z1"
                ],
                "expected": "Structured data from Neo4j database"
            },
            {
                "category": "📄 RAG Semantic Search",
                "queries": [
                    "Find maintenance documentation",
                    "Search for sensor information",
                    "What data do we have about operators?",
                    "Look for production batch details"
                ],
                "expected": "Relevant documents from CSV data"
            }
        ]
        
        if self.openai_available:
            examples.append({
                "category": "🤖 OpenAI Enhanced Queries",
                "queries": [
                    "enhanced What's wrong with machine M100 and how do I fix it?",
                    "simple Why do machines vibrate too much?",
                    "technical Provide detailed analysis of sensor readings",
                    "executive Give me a summary of facility issues"
                ],
                "expected": "AI-generated explanations and recommendations"
            })
        
        for example in examples:
            print(f"\n{example['category']}")
            print("-" * 50)
            for query in example['queries']:
                print(f"  • {query}")
            print(f"Expected: {example['expected']}")
    
    async def process_query(self, query: str):
        """Process a user query"""
        if not query.strip():
            return
        
        print(f"\n{'='*60}")
        print(f"🔍 Processing: {query}")
        print('='*60)
        
        # Parse command
        use_enhanced = False
        response_mode = "balanced"
        preferred_agent = None
        
        # Check for command prefixes
        if query.startswith('rag '):
            preferred_agent = 'rag'
            query = query[4:].strip()
        elif query.startswith('graph '):
            preferred_agent = 'graph'
            query = query[6:].strip()
        elif query.startswith('enhanced '):
            use_enhanced = True
            query = query[9:].strip()
        elif query.startswith('simple '):
            use_enhanced = True
            response_mode = "simple"
            query = query[7:].strip()
        elif query.startswith('technical '):
            use_enhanced = True
            response_mode = "technical"
            query = query[10:].strip()
        elif query.startswith('executive '):
            use_enhanced = True
            response_mode = "executive"
            query = query[10:].strip()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process the query
            if use_enhanced and self.openai_available:
                result = await self.system.process_enhanced_query(
                    query, 
                    use_llm=True, 
                    response_mode=response_mode
                )
            else:
                result = await self.system.process_query(query, preferred_agent)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Display results
            success = result.get('success', False)
            
            if success:
                # Show main response content first
                content = result.get('content')
                if content:
                    print(content)
                
                # Show OpenAI response if available and no content shown yet
                if not content:
                    llm_response = result.get('llm_response')
                    if llm_response and llm_response.get('success'):
                        print(llm_response['content'])


            
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    async def run(self):
        """Run the interactive interface"""
        # Initialize system silently
        success = await self.initialize()
        if not success:
            return
        
        try:
            while True:
                try:
                    query = input("Your query: ").strip()
                    
                    if not query:
                        continue
                    
                    # Handle system commands
                    if query.lower() == 'quit':
                        print("\n👋 Goodbye!")
                        break
                    elif query.lower() == 'help':
                        await self.show_help()
                        continue
                    elif query.lower() == 'examples':
                        await self.show_examples()
                        continue
                    elif query.lower() == 'status':
                        await self.system.show_system_status()
                        continue
                    elif query.lower() == 'stats':
                        stats = self.system.get_quick_stats()
                        print(f"\n📊 Usage Statistics:")
                        print(f"Total requests: {stats['total_requests']}")
                        print(f"Success rate: {stats['success_rate']*100:.1f}%")
                        for agent, count in stats['agent_usage'].items():
                            if count > 0:
                                print(f"{agent.upper()}: {count} requests")
                        continue
                    
                    # Process the query
                    await self.process_query(query)
                
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
        
        finally:
            if self.system:
                await self.system.cleanup()

async def main():
    """Main entry point"""
    interface = InteractiveQueryInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main())