"""
ACP (AI Control Protocol) Integration
Intelligent agent coordination and decision-making using LLM

This module provides:
1. AI-driven tool selection and routing
2. Multi-step reasoning and planning
3. Dynamic agent coordination
4. Context-aware decision making
5. Integration with MCP servers and OpenAI function calling
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, date
import decimal

from ai_brain import OpenAIIntegration, EnhancedRAGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling datetime and other non-serializable types"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)

class AgentCapability(Enum):
    """Available agent capabilities"""
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_QUERY = "graph_query"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    SYSTEM_STATS = "system_stats"
    CONTEXT_GENERATION = "context_generation"
    DATA_SYNTHESIS = "data_synthesis"

@dataclass
class ToolDescriptor:
    """Describes an available tool for LLM function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    agent_type: str
    mcp_tool: Optional[str] = None
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@dataclass
class ACPRequest:
    """ACP request with context and metadata"""
    query: str
    user_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    preferred_agents: Optional[List[str]] = None
    response_format: str = "comprehensive"
    
@dataclass
class ACPResponse:
    """ACP response with decision trail and results"""
    success: bool
    final_answer: str
    decision_trail: List[Dict[str, Any]]
    tools_used: List[str]
    agents_involved: List[str]
    execution_time: float
    confidence_score: Optional[float] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None

class ACPCoordinator:
    """AI Control Protocol Coordinator - The AI Brain"""
    
    def __init__(self):
        self.openai_integration = None
        self.available_tools = {}
        self.registered_agents = {}
        self.decision_history = []
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_decision_time': 0.0,
            'tool_usage_stats': {},
            'agent_usage_stats': {}
        }
    
    async def initialize(self) -> bool:
        """Initialize ACP coordinator"""
        try:
            logger.info("🧠 Initializing ACP Coordinator...")
            
            # Initialize OpenAI integration
            self.openai_integration = OpenAIIntegration()
            # Note: OpenAI integration doesn't need async initialize, constructor handles it
            
            # Check if OpenAI is available
            if not self.openai_integration.is_available():
                logger.warning("⚠️ OpenAI not available, ACP will use fallback methods")
            
            # Register default tools
            await self._register_default_tools()
            
            logger.info("✅ ACP Coordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ ACP initialization failed: {e}")
            return False
    
    async def _register_default_tools(self):
        """Register default system tools"""
        
        # RAG/Semantic Search Tools
        self.register_tool(ToolDescriptor(
            name="search_documents",
            description="Search shopfloor documents using semantic similarity. Use for finding information about maintenance, procedures, operators, or any document-based knowledge.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant documents"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            agent_type="rag",
            mcp_tool="search_documents"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_document_context",
            description="Get formatted context from documents for detailed analysis. Use when you need comprehensive background information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to generate context for"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of documents to use for context (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            },
            agent_type="rag",
            mcp_tool="get_context"
        ))
        
        # Graph Database Tools
        self.register_tool(ToolDescriptor(
            name="find_high_vibration_machines",
            description="Find machines with vibration readings above a specified threshold. Use for identifying maintenance issues.",
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Vibration threshold in mm/s (default: 7.5)",
                        "default": 7.5
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (default: mm/s)",
                        "default": "mm/s"
                    }
                }
            },
            agent_type="graph",
            mcp_tool="highVibrationMachines"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_overdue_work_orders",
            description="Retrieve all overdue work orders. Use for maintenance planning and urgent task identification.",
            parameters={
                "type": "object",
                "properties": {}
            },
            agent_type="graph",
            mcp_tool="overdueWorkOrders"
        ))
        
        self.register_tool(ToolDescriptor(
            name="find_machine_operator",
            description="Find the current operator of a specific machine. Use for personnel and responsibility queries.",
            parameters={
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string",
                        "description": "Machine ID (e.g., M100, M200)"
                    }
                },
                "required": ["machine_id"]
            },
            agent_type="graph",
            mcp_tool="currentOperator"
        ))
        
        self.register_tool(ToolDescriptor(
            name="list_sensors_by_zone",
            description="List all sensors in a specific facility zone. Use for zone-based monitoring queries.",
            parameters={
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "string",
                        "description": "Zone ID (e.g., Z1, Z2)"
                    }
                },
                "required": ["zone_id"]
            },
            agent_type="graph",
            mcp_tool="sensorsByZone"
        ))
        
        self.register_tool(ToolDescriptor(
            name="find_machines_due_maintenance",
            description="Find machines due for maintenance by a certain date. Use for preventive maintenance planning.",
            parameters={
                "type": "object",
                "properties": {
                    "cutoff_date": {
                        "type": "string",
                        "description": "Cutoff date in ISO format (YYYY-MM-DD)"
                    }
                },
                "required": ["cutoff_date"]
            },
            agent_type="graph",
            mcp_tool="dueForMaintenance"
        ))
        
        self.register_tool(ToolDescriptor(
            name="list_all_machines",
            description="List all available machines in the shopfloor with their basic information. Use when users ask about what machines are available, all machines, or machine inventory.",
            parameters={
                "type": "object",
                "properties": {}
            },
            agent_type="graph",
            mcp_tool="listAllMachines"
        ))
        
        self.register_tool(ToolDescriptor(
            name="find_qualified_operators",
            description="Find operators with specific certifications for maintenance or operation tasks. Use when assigning work to qualified personnel.",
            parameters={
                "type": "object",
                "properties": {
                    "certification": {
                        "type": "string",
                        "description": "The certification or skill required (e.g., 'CNC-1', 'Safety-A')"
                    }
                },
                "required": ["certification"]
            },
            agent_type="graph",
            mcp_tool="findQualifiedOperators"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_work_orders_by_status",
            description="Find work orders by their current status (Overdue, InProgress, Scheduled). Use for workflow management and tracking.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string", 
                        "description": "Work order status: 'Overdue', 'InProgress', or 'Scheduled'"
                    }
                },
                "required": ["status"]
            },
            agent_type="graph",
            mcp_tool="workOrdersByStatus"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_maintenance_history",
            description="Get detailed maintenance history and logs for a specific machine. Use for understanding past issues and repair patterns.",
            parameters={
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string",
                        "description": "The machine ID to get history for"
                    }
                },
                "required": ["machine_id"]
            },
            agent_type="graph",
            mcp_tool="maintenanceHistory"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_machine_current_sensors",
            description="Get all current sensor readings for a specific machine. Use for real-time monitoring and diagnostics.",
            parameters={
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string",
                        "description": "The machine ID to get sensor data for"
                    }
                },
                "required": ["machine_id"]
            },
            agent_type="graph",
            mcp_tool="machineCurrentSensors"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_current_production_batches",
            description="Find currently active production batches. Use for production tracking and scheduling.",
            parameters={
                "type": "object",
                "properties": {}
            },
            agent_type="graph",
            mcp_tool="currentProductionBatches"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_machines_in_zone",
            description="List all machines located in a specific zone. Use for zone-based operations and management.",
            parameters={
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "string",
                        "description": "The zone ID to search in"
                    }
                },
                "required": ["zone_id"]
            },
            agent_type="graph",
            mcp_tool="machinesInZone"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_operators_by_shift",
            description="List all operators working a specific shift. Use for shift planning and operator assignment.",
            parameters={
                "type": "object",
                "properties": {
                    "shift": {
                        "type": "string",
                        "description": "The shift name: 'Day' or 'Night'"
                    }
                },
                "required": ["shift"]
            },
            agent_type="graph",
            mcp_tool="operatorsByShift"
        ))
        
        self.register_tool(ToolDescriptor(
            name="get_temperature_alerts",
            description="Find machines with high temperature sensor readings above threshold. Use for equipment monitoring and preventing overheating.",
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Temperature threshold in Celsius (default: 60.0)"
                    }
                },
                "required": ["threshold"]
            },
            agent_type="graph",
            mcp_tool="temperatureAlerts"
        ))
        
        # System Tools
        self.register_tool(ToolDescriptor(
            name="get_system_statistics",
            description="Get comprehensive system statistics and health information. Use for system monitoring queries.",
            parameters={
                "type": "object",
                "properties": {}
            },
            agent_type="system",
            mcp_tool="get_system_stats"
        ))
    
    def register_tool(self, tool: ToolDescriptor):
        """Register a tool for LLM usage"""
        self.available_tools[tool.name] = tool
        logger.info(f"🔧 Registered tool: {tool.name}")
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent instance"""
        self.registered_agents[agent_name] = agent_instance
        logger.info(f"🤖 Registered agent: {agent_name}")
    
    async def process_request(self, request: ACPRequest) -> ACPResponse:
        """Main ACP processing - AI-driven decision making"""
        start_time = time.time()
        decision_trail = []
        tools_used = []
        agents_involved = []
        
        try:
            logger.info(f"🧠 ACP processing request: {request.query}")
            
            # Step 1: AI Planning Phase
            planning_result = await self._ai_planning_phase(request, decision_trail)
            if not planning_result['success']:
                return ACPResponse(
                    success=False,
                    final_answer="",
                    decision_trail=decision_trail,
                    tools_used=tools_used,
                    agents_involved=agents_involved,
                    execution_time=time.time() - start_time,
                    error=planning_result.get('error')
                )
            
            # Step 2: Tool Execution Phase
            execution_results = await self._tool_execution_phase(
                planning_result['execution_plan'], 
                decision_trail,
                tools_used,
                agents_involved
            )
            
            # Step 3: AI Synthesis Phase
            final_answer = await self._ai_synthesis_phase(
                request,
                execution_results,
                decision_trail
            )
            
            # Update statistics
            self._update_stats(True, time.time() - start_time, tools_used, agents_involved)
            
            return ACPResponse(
                success=True,
                final_answer=final_answer,
                decision_trail=decision_trail,
                tools_used=tools_used,
                agents_involved=agents_involved,
                execution_time=time.time() - start_time,
                confidence_score=execution_results.get('confidence_score'),
                suggestions=execution_results.get('suggestions')
            )
            
        except Exception as e:
            logger.error(f"❌ ACP processing error: {e}")
            self._update_stats(False, time.time() - start_time, tools_used, agents_involved)
            
            return ACPResponse(
                success=False,
                final_answer="",
                decision_trail=decision_trail,
                tools_used=tools_used,
                agents_involved=agents_involved,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _ai_planning_phase(self, request: ACPRequest, decision_trail: List) -> Dict[str, Any]:
        """AI decides what tools to use and in what order"""
        
        decision_trail.append({
            "phase": "planning",
            "timestamp": time.time(),
            "action": "AI analyzing request and selecting tools"
        })
        
        # Prepare function descriptions for OpenAI
        functions = [tool.to_openai_function() for tool in self.available_tools.values()]
        
        # Create planning prompt
        planning_prompt = f"""
You are an AI coordinator for a shopfloor data analysis system. 

User Query: "{request.query}"

Available tools and their purposes:
{self._format_tools_for_prompt()}

Your task:
1. Analyze the user's request
2. Decide which tools to use and in what order
3. Explain your reasoning

Respond with a JSON object containing:
- "reasoning": Your step-by-step thinking
- "tools_needed": List of tool names to use
- "execution_order": Order of execution
- "expected_outcome": What you expect to achieve

Be strategic - you can use multiple tools and combine their results.
"""
        
        try:
            response = await self.openai_integration.generate_response(
                query=planning_prompt,
                context="",
                response_mode="technical",
                use_functions=False  # We want reasoning, not function calls
            )
            
            if response.success:
                # Try to parse the AI's planning response
                try:
                    plan_data = json.loads(response.content)
                    decision_trail.append({
                        "phase": "planning", 
                        "result": plan_data,
                        "timestamp": time.time()
                    })
                    return {
                        "success": True,
                        "execution_plan": plan_data
                    }
                except json.JSONDecodeError:
                    # Fallback to text-based planning
                    decision_trail.append({
                        "phase": "planning",
                        "result": {"reasoning": response.content},
                        "timestamp": time.time()
                    })
                    
                    # Use function calling as fallback
                    return await self._fallback_function_calling(request, decision_trail)
            else:
                return {"success": False, "error": response.error}
                
        except Exception as e:
            logger.error(f"Planning phase error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_function_calling(self, request: ACPRequest, decision_trail: List) -> Dict[str, Any]:
        """Use OpenAI function calling for intelligent tool selection"""
        
        decision_trail.append({
            "phase": "ai_function_calling",
            "action": "Using OpenAI function calling for intelligent tool selection",
            "timestamp": time.time()
        })
        
        functions = [tool.to_openai_function() for tool in self.available_tools.values()]
        
        try:
            # Use enhanced OpenAI function calling from our integration
            response = await self.openai_integration.generate_response_with_functions(
                query=request.query,
                context="",
                response_mode=request.response_format,
                available_functions=functions,
                function_executor=self._execute_single_tool,
                system_prompt="You are an intelligent shopfloor data analysis assistant. Use the available tools to provide comprehensive answers to user questions. You can call multiple tools if needed to gather complete information."
            )
            
            if response.success:
                if response.function_calls:
                    # AI made function calls
                    execution_plan = {
                        "reasoning": "AI intelligently selected and executed tools",
                        "tools_needed": [call["name"] for call in response.function_calls],
                        "function_calls": response.function_calls,
                        "ai_synthesis": response.content
                    }
                    
                    decision_trail.append({
                        "phase": "ai_function_execution",
                        "result": f"Executed {len(response.function_calls)} tools successfully",
                        "tools": [call["name"] for call in response.function_calls],
                        "timestamp": time.time()
                    })
                    
                    return {
                        "success": True,
                        "execution_plan": execution_plan
                    }
                else:
                    # AI provided direct response without tools
                    return {
                        "success": True,
                        "execution_plan": {
                            "reasoning": "AI determined no specific tools needed",
                            "tools_needed": [],
                            "direct_response": response.content
                        }
                    }
            else:
                logger.error(f"OpenAI function calling failed: {response.error}")
                return {"success": False, "error": response.error}
                
        except Exception as e:
            logger.error(f"Function calling error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _tool_execution_phase(self, execution_plan: Dict, decision_trail: List, tools_used: List, agents_involved: List) -> Dict[str, Any]:
        """Execute the planned tools"""
        
        results = {}
        
        # Handle direct response case
        if "direct_response" in execution_plan:
            return {
                "direct_response": execution_plan["direct_response"],
                "tool_results": {}
            }
        
        # Execute function calls if available
        if "function_calls" in execution_plan:
            for func_call in execution_plan["function_calls"]:
                tool_name = func_call["name"]
                arguments = func_call["arguments"]
                
                decision_trail.append({
                    "phase": "execution",
                    "action": f"Executing tool: {tool_name}",
                    "arguments": arguments,
                    "timestamp": time.time()
                })
                
                result = await self._execute_single_tool(tool_name, arguments)
                results[tool_name] = result
                tools_used.append(tool_name)
                
                if tool_name in self.available_tools:
                    agent_type = self.available_tools[tool_name].agent_type
                    if agent_type not in agents_involved:
                        agents_involved.append(agent_type)
        
        # Execute tools from tools_needed list (legacy format)
        elif "tools_needed" in execution_plan:
            for tool_name in execution_plan["tools_needed"]:
                # For now, execute with default parameters
                # TODO: Extract parameters from reasoning or use smart defaults
                result = await self._execute_single_tool(tool_name, {})
                results[tool_name] = result
                tools_used.append(tool_name)
        
        return {
            "tool_results": results,
            "execution_summary": f"Executed {len(results)} tools successfully"
        }
    
    async def _execute_single_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool using registered agents or MCP if available"""
        
        if tool_name not in self.available_tools:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        tool = self.available_tools[tool_name]
        agent_type = tool.agent_type
        
        try:
            # Try MCP server first if available
            if tool.mcp_tool and await self._try_mcp_execution(tool_name, tool.mcp_tool, arguments):
                return await self._try_mcp_execution(tool_name, tool.mcp_tool, arguments)
            
            # Fallback to direct agent execution
            if agent_type == "rag" and "rag" in self.registered_agents:
                agent = self.registered_agents["rag"]
                if tool.mcp_tool == "search_documents":
                    return await agent.search_documents(
                        arguments.get("query", ""),
                        k=arguments.get("k", 5)
                    )
                elif tool.mcp_tool == "get_context":
                    return await agent.get_context(
                        arguments.get("query", ""),
                        k=arguments.get("k", 3)
                    )
            
            elif agent_type == "graph" and "graph" in self.registered_agents:
                agent = self.registered_agents["graph"]
                if tool.mcp_tool == "highVibrationMachines":
                    return await agent.find_high_vibration_machines(
                        threshold=arguments.get("threshold", 7.5),
                        unit=arguments.get("unit", "mm/s")
                    )
                elif tool.mcp_tool == "overdueWorkOrders":
                    return await agent.get_overdue_work_orders()
                elif tool.mcp_tool == "currentOperator":
                    return await agent.get_current_operator(
                        arguments.get("machine_id", "")
                    )
                elif tool.mcp_tool == "sensorsByZone":
                    return await agent.get_sensors_by_zone(
                        arguments.get("zone_id", "")
                    )
                elif tool.mcp_tool == "dueForMaintenance":
                    return await agent.get_machines_due_maintenance(
                        arguments.get("cutoff_date", "")
                    )
                elif tool.mcp_tool == "listAllMachines":
                    return await agent.list_all_machines()
                elif tool.mcp_tool == "findQualifiedOperators":
                    return await agent.find_qualified_operators(
                        arguments.get("certification", "")
                    )
                elif tool.mcp_tool == "workOrdersByStatus":
                    return await agent.get_work_orders_by_status(
                        arguments.get("status", "")
                    )
                elif tool.mcp_tool == "maintenanceHistory":
                    return await agent.get_maintenance_history(
                        arguments.get("machine_id", "")
                    )
                elif tool.mcp_tool == "machineCurrentSensors":
                    return await agent.get_machine_current_sensors(
                        arguments.get("machine_id", "")
                    )
                elif tool.mcp_tool == "currentProductionBatches":
                    return await agent.get_current_production_batches()
                elif tool.mcp_tool == "machinesInZone":
                    return await agent.get_machines_in_zone(
                        arguments.get("zone_id", "")
                    )
                elif tool.mcp_tool == "operatorsByShift":
                    return await agent.get_operators_by_shift(
                        arguments.get("shift", "")
                    )
                elif tool.mcp_tool == "temperatureAlerts":
                    return await agent.get_temperature_alerts(
                        arguments.get("threshold", 60.0)
                    )
            
            return {"success": False, "error": f"Agent {agent_type} not available or tool not implemented"}
            
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _try_mcp_execution(self, tool_name: str, mcp_tool: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to execute tool via MCP server if available"""
        try:
            # Check for available MCP client connections
            # For now, this is a placeholder - would need actual MCP client implementation
            # Return None to indicate MCP not available, fallback to direct agent calls
            return None
            
        except Exception as e:
            logger.debug(f"MCP execution failed for {tool_name}: {e}")
            return None
    
    async def _ai_synthesis_phase(self, request: ACPRequest, execution_results: Dict, decision_trail: List) -> str:
        """AI synthesizes final answer from tool results"""
        
        decision_trail.append({
            "phase": "synthesis",
            "action": "AI synthesizing final answer from tool results",
            "timestamp": time.time()
        })
        
        # Handle direct response
        if "direct_response" in execution_results:
            return execution_results["direct_response"]
        
        # Format tool results for synthesis
        results_summary = self._format_results_for_synthesis(execution_results.get("tool_results", {}))
        
        synthesis_prompt = f"""
Based on the user's question: "{request.query}"

And the following tool results:
{results_summary}

Provide a comprehensive, helpful answer that:
1. Directly addresses the user's question
2. Uses the data from the tool results
3. Provides actionable insights when appropriate
4. Is clear and well-structured

Format your response in a natural, conversational way.
"""
        
        try:
            response = await self.openai_integration.generate_response(
                query=synthesis_prompt,
                context="",
                response_mode=request.response_format
            )
            
            if response.success:
                decision_trail.append({
                    "phase": "synthesis",
                    "result": "Successfully synthesized final answer",
                    "timestamp": time.time()
                })
                return response.content
            else:
                return f"I found the following information: {results_summary}"
                
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"I found the following information: {results_summary}"
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools for AI prompt"""
        tool_descriptions = []
        for tool in self.available_tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)
    
    def _format_results_for_synthesis(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results for synthesis prompt"""
        formatted_results = []
        for tool_name, result in tool_results.items():
            if result.get('success'):
                data = result.get('data', result)
                formatted_results.append(f"{tool_name}: {json.dumps(data, default=safe_json_serialize)}")
            else:
                formatted_results.append(f"{tool_name}: Error - {result.get('error', 'Unknown error')}")
        
        return "\n".join(formatted_results)
    
    def _update_stats(self, success: bool, execution_time: float, tools_used: List[str], agents_involved: List[str]):
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        if success:
            self.performance_stats['successful_requests'] += 1
        
        # Update average decision time
        total_time = self.performance_stats['average_decision_time'] * (self.performance_stats['total_requests'] - 1)
        self.performance_stats['average_decision_time'] = (total_time + execution_time) / self.performance_stats['total_requests']
        
        # Update tool usage stats
        for tool in tools_used:
            self.performance_stats['tool_usage_stats'][tool] = \
                self.performance_stats['tool_usage_stats'].get(tool, 0) + 1
        
        # Update agent usage stats
        for agent in agents_involved:
            self.performance_stats['agent_usage_stats'][agent] = \
                self.performance_stats['agent_usage_stats'].get(agent, 0) + 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return self.decision_history[-limit:]

# Test function
async def test_acp_coordinator():
    """Test the ACP coordinator"""
    coordinator = ACPCoordinator()
    
    if await coordinator.initialize():
        print("✅ ACP Coordinator initialized successfully")
        
        # Test request
        test_request = ACPRequest(
            query="What machines have high vibration and need maintenance?",
            response_format="comprehensive"
        )
        
        response = await coordinator.process_request(test_request)
        
        print(f"Success: {response.success}")
        print(f"Answer: {response.final_answer}")
        print(f"Tools Used: {response.tools_used}")
        print(f"Execution Time: {response.execution_time:.3f}s")
    else:
        print("❌ ACP Coordinator initialization failed")

if __name__ == "__main__":
    asyncio.run(test_acp_coordinator())