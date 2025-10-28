"""
Enhanced AI System Test
Tests the new LLM-driven tool selection with ACP

This script tests:
1. ACP (AI Control Protocol) intelligent coordination
2. OpenAI Function Calling for tool selection
3. Multi-step reasoning capabilities
4. Enhanced decision-making vs rule-based routing
"""

import asyncio
import logging
import time
from ai_system import WFCIntegrationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSystemTester:
    """Test the enhanced AI-driven system"""
    
    def __init__(self):
        self.system = None
        self.test_results = []
    
    async def initialize(self):
        """Initialize the enhanced system"""
        print("🧠 Initializing Enhanced AI-Driven System for Testing...")
        print("=" * 60)
        
        # Initialize with enhanced mode enabled
        self.system = WFCIntegrationSystem(use_enhanced_mode=True)
        success = await self.system.initialize()
        
        if success:
            print("✅ Enhanced System Initialized Successfully!")
            await self.system.show_system_status()
            return True
        else:
            print("❌ Enhanced System Initialization Failed")
            return False
    
    async def test_ai_decision_making(self):
        """Test AI-driven decision making vs rule-based"""
        print("\n🧠 TESTING AI-DRIVEN DECISION MAKING")
        print("=" * 50)
        
        test_queries = [
            {
                "query": "Who is Nick and what is he working on?",
                "description": "Complex query requiring multi-step reasoning",
                "expected_reasoning": "AI should search for Nick, then find his work details"
            },
            {
                "query": "What machines have high vibration and need maintenance?",
                "description": "Hybrid query requiring both graph data and analysis",
                "expected_reasoning": "AI should query graph DB then provide maintenance insights"
            },
            {
                "query": "Explain the relationship between sensor readings and machine performance",
                "description": "Complex analytical query requiring synthesis",
                "expected_reasoning": "AI should gather sensor data and provide analytical explanation"
            },
            {
                "query": "Find overdue work orders and prioritize them by urgency",
                "description": "Query requiring data retrieval and intelligent prioritization",
                "expected_reasoning": "AI should get overdue orders then apply prioritization logic"
            }
        ]
        
        for test in test_queries:
            print(f"\n🔍 Testing: {test['query']}")
            print(f"Expected: {test['expected_reasoning']}")
            print("-" * 40)
            
            start_time = time.time()
            
            # Test with Enhanced AI System (ACP)
            result = await self.system.process_acp_query(
                query=test['query'],
                response_mode="technical"
            )
            
            execution_time = time.time() - start_time
            
            if result['success']:
                print(f"✅ AI Decision Success - {execution_time:.3f}s")
                print(f"🤖 Agent Used: {result['agent_used']}")
                
                # Show AI reasoning if available
                decision_trail = result['data'].get('decision_trail', '')
                if decision_trail:
                    print(f"🧠 AI Reasoning: {decision_trail}")
                
                # Show tools used
                tools_used = result['data'].get('tools_executed', [])
                if tools_used:
                    print(f"🔧 Tools Used: {', '.join(tools_used)}")
                
                # Show confidence if available
                confidence = result['data'].get('ai_confidence')
                if confidence:
                    print(f"📊 AI Confidence: {confidence}")
                
                # Show response preview
                response_preview = result['data'].get('ai_response', '')[:200]
                print(f"📝 Response: {response_preview}...")
                
            else:
                print(f"❌ AI Decision Failed: {result.get('error')}")
            
            self.test_results.append({
                'query': test['query'],
                'success': result['success'],
                'execution_time': execution_time,
                'agent_used': result.get('agent_used'),
                'error': result.get('error')
            })
            
            print()
    
    async def test_function_calling_intelligence(self):
        """Test OpenAI Function Calling integration"""
        print("\n🔧 TESTING FUNCTION CALLING INTELLIGENCE")
        print("=" * 50)
        
        intelligent_queries = [
            {
                "query": "I need a comprehensive report on machine M100 - check its vibration, operator, and maintenance status",
                "description": "Should trigger multiple function calls in sequence"
            },
            {
                "query": "What's the operational status of zone Z1 and what actions are needed?", 
                "description": "Should query sensors, then provide actionable recommendations"
            },
            {
                "query": "Compare the maintenance needs across all machines and suggest priorities",
                "description": "Should gather maintenance data then synthesize priorities"
            }
        ]
        
        for test in intelligent_queries:
            print(f"\n🎯 Testing: {test['query']}")
            print(f"Purpose: {test['description']}")
            print("-" * 40)
            
            start_time = time.time()
            
            result = await self.system.process_enhanced_query(
                query=test['query'],
                use_llm=True,
                response_mode="comprehensive"
            )
            
            execution_time = time.time() - start_time
            
            if result['success']:
                print(f"✅ Function Calling Success - {execution_time:.3f}s")
                
                # Show function calls made
                function_calls = result['data'].get('function_calls', [])
                if function_calls:
                    print(f"🔧 Function Calls Made: {len(function_calls)}")
                    for call in function_calls:
                        call_name = call.get('phase', call.get('action', 'Unknown'))
                        print(f"   • {call_name}")
                
                # Show AI synthesis
                ai_response = result['data'].get('content', '')
                if ai_response:
                    print(f"🤖 AI Synthesis: {ai_response[:150]}...")
                
            else:
                print(f"❌ Function Calling Failed: {result.get('error')}")
            
            print()
    
    async def test_multi_step_reasoning(self):
        """Test complex multi-step reasoning capabilities"""
        print("\n🔗 TESTING MULTI-STEP REASONING")
        print("=" * 50)
        
        complex_scenarios = [
            {
                "query": "What are the top 3 most critical issues in the facility right now and what should we do about them?",
                "steps": ["Identify critical issues", "Prioritize by severity", "Provide action plans"]
            },
            {
                "query": "Analyze the relationship between operator performance and machine maintenance schedules",
                "steps": ["Get operator data", "Get maintenance schedules", "Analyze correlations", "Provide insights"]
            },
            {
                "query": "Create a maintenance strategy for the next month based on current sensor readings and work order history",
                "steps": ["Analyze sensor trends", "Review work order patterns", "Predict maintenance needs", "Create strategy"]
            }
        ]
        
        for scenario in complex_scenarios:
            print(f"\n🎯 Complex Scenario: {scenario['query']}")
            print(f"Expected Steps: {' → '.join(scenario['steps'])}")
            print("-" * 40)
            
            start_time = time.time()
            
            result = await self.system.process_acp_query(
                query=scenario['query'],
                response_mode="executive"
            )
            
            execution_time = time.time() - start_time
            
            if result['success']:
                print(f"✅ Multi-Step Reasoning Success - {execution_time:.3f}s")
                
                # Show the reasoning trail
                decision_trail = result['data'].get('decision_trail', '')
                if decision_trail:
                    print(f"🧠 AI Reasoning Trail: {decision_trail}")
                
                # Show final synthesis
                final_answer = result['data'].get('ai_response', '')
                if final_answer:
                    print(f"📋 Executive Summary: {final_answer[:200]}...")
                
                # Show suggestions if any
                suggestions = result.get('ai_suggestions', [])
                if suggestions:
                    print(f"💡 AI Suggestions: {', '.join(suggestions[:3])}")
                
            else:
                print(f"❌ Multi-Step Reasoning Failed: {result.get('error')}")
            
            print()
    
    async def compare_ai_vs_rule_based(self):
        """Compare AI-driven vs rule-based routing"""
        print("\n⚔️ COMPARING AI-DRIVEN VS RULE-BASED ROUTING")
        print("=" * 50)
        
        comparison_queries = [
            "What's wrong with the machines and how do I fix them?",
            "Show me everything about maintenance operations",
            "I need help understanding the facility status"
        ]
        
        for query in comparison_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 30)
            
            # Test AI-Driven Approach
            start_time = time.time()
            ai_result = await self.system.process_acp_query(query, response_mode="balanced")
            ai_time = time.time() - start_time
            
            # Test Standard Approach
            start_time = time.time()
            standard_result = await self.system.process_query(query)
            standard_time = time.time() - start_time
            
            print(f"🧠 AI-Driven: {'✅' if ai_result['success'] else '❌'} ({ai_time:.3f}s)")
            if ai_result['success']:
                print(f"   Agent: {ai_result['agent_used']}, Tools: {len(ai_result['data'].get('tools_executed', []))}")
            
            print(f"📋 Rule-Based: {'✅' if standard_result['success'] else '❌'} ({standard_time:.3f}s)")
            if standard_result['success']:
                print(f"   Agent: {standard_result['agent_used']}")
            
            print()
    
    async def show_test_summary(self):
        """Show comprehensive test summary"""
        print("\n📊 ENHANCED AI SYSTEM TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        avg_time = sum(result['execution_time'] for result in self.test_results) / max(1, total_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success Rate: {successful_tests/max(1, total_tests)*100:.1f}%")
        print(f"Average Time: {avg_time:.3f}s")
        
        # Show system stats
        stats = self.system.get_quick_stats()
        print(f"\nSystem Mode: {stats.get('system_mode', 'Unknown')}")
        print(f"AI Usage Rate: {stats.get('ai_usage_rate', 0)*100:.1f}%")
        print(f"ACP Usage Rate: {stats.get('acp_usage_rate', 0)*100:.1f}%")
        
        # Agent usage breakdown
        print(f"\n🤖 Agent Usage:")
        for agent, count in stats['agent_usage'].items():
            if count > 0:
                print(f"   {agent.upper()}: {count} requests")
        
        print("\n🎉 Enhanced AI System Testing Complete!")

async def main():
    """Main test runner"""
    tester = EnhancedSystemTester()
    
    if await tester.initialize():
        print("\n🚀 Starting Enhanced AI System Tests...")
        
        # Run all test suites
        await tester.test_ai_decision_making()
        await tester.test_function_calling_intelligence()
        await tester.test_multi_step_reasoning()
        await tester.compare_ai_vs_rule_based()
        
        # Show final results
        await tester.show_test_summary()
        
    else:
        print("❌ Cannot run tests - system initialization failed")

if __name__ == "__main__":
    asyncio.run(main())