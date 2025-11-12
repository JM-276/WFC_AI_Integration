"""
Workflow Canvas Integration Module
Provides endpoints and utilities for integrating WFC AI system with Siemens Workflow Canvas
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import logging
from datetime import datetime

from ai_system import WFCIntegrationSystem
from ai_coordinator import ACPRequest, ACPResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Workflow Canvas Request Models
class WorkflowCanvasRequest(BaseModel):
    """Request model matching Workflow Canvas HTTP Server (RESTful.1) format"""
    url: Optional[str] = "/"
    Body: Optional[Dict[str, Any]] = None
    Params: Optional[Dict[str, str]] = None
    Headers: Optional[Dict[str, str]] = None
    Method: Optional[str] = "POST"


class WorkflowCanvasResponse(BaseModel):
    """Response model for Workflow Canvas RESTful.1 output"""
    Code: int
    Data: Dict[str, Any]
    Status: str
    Server: str = "WFC-AI-Integration"
    URL: str
    Method: str
    Headers: Dict[str, str]
    Body: Dict[str, Any]
    Timeout: int = 5


class WorkflowCanvasIntegration:
    """Integration layer between Workflow Canvas and WFC AI System"""
    
    def __init__(self, wfc_system: WFCIntegrationSystem):
        self.wfc_system = wfc_system
        self.request_count = 0
        self.success_count = 0
        
    async def process_workflow_request(
        self, 
        workflow_request: WorkflowCanvasRequest
    ) -> WorkflowCanvasResponse:
        """
        Process incoming request from Workflow Canvas RESTful.1 block
        
        Expected Body format:
        {
            "query": "What machines need maintenance?",
            "mode": "enhanced",  # optional: enhanced, rag, graph
            "response_style": "balanced",  # optional: simple, balanced, technical, executive
            "max_results": 5
        }
        """
        self.request_count += 1
        start_time = datetime.now()
        
        try:
            # Extract request body
            body = workflow_request.Body or {}
            query = body.get("query")
            
            if not query:
                raise ValueError("Missing required field: 'query' in request body")
            
            # Extract optional parameters
            mode = body.get("mode", "auto")  # auto, enhanced, rag, graph
            response_style = body.get("response_style", "balanced")
            max_results = body.get("max_results", 5)
            
            logger.info(f"📥 Workflow Canvas Request #{self.request_count}: {query}")
            
            # Process based on mode
            if mode == "enhanced":
                result = await self.wfc_system.process_enhanced_query(
                    query=query,
                    use_llm=True,
                    response_mode=response_style,
                    max_results=max_results
                )
            elif mode == "rag":
                result = await self.wfc_system.process_query(
                    query=query,
                    preferred_agent="rag",
                    use_llm=True,
                    response_mode=response_style,
                    max_results=max_results
                )
            elif mode == "graph":
                result = await self.wfc_system.process_query(
                    query=query,
                    preferred_agent="graph",
                    use_llm=True,
                    response_mode=response_style,
                    max_results=max_results
                )
            else:  # auto mode
                result = await self.wfc_system.process_query(
                    query=query,
                    use_llm=True,
                    response_mode=response_style,
                    max_results=max_results
                )
            
            # Check if processing was successful
            if result.get("success"):
                self.success_count += 1
                status = "Good"
                code = 0
            else:
                status = "Error"
                code = 1
            
            # Build response data
            response_data = {
                "answer": result.get("content") or result.get("llm_response", {}).get("content", ""),
                "agent_used": result.get("agent_used", "unknown"),
                "execution_time": result.get("execution_time", 0),
                "success": result.get("success", False),
                "metadata": {
                    "decision_reasoning": result.get("decision_reasoning"),
                    "confidence_score": result.get("confidence_score"),
                    "sources_used": result.get("sources_used", []),
                    "tools_used": result.get("tools_used", [])
                }
            }
            
            # Create Workflow Canvas compatible response
            wfc_response = WorkflowCanvasResponse(
                Code=code,
                Data=response_data,
                Status=status,
                URL=workflow_request.url or "/",
                Method=workflow_request.Method or "POST",
                Headers=workflow_request.Headers or {},
                Body=response_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Workflow Canvas Response #{self.request_count}: {status} ({execution_time:.2f}s)")
            
            return wfc_response
            
        except Exception as e:
            logger.error(f"❌ Workflow Canvas processing error: {e}")
            
            # Return error response in Workflow Canvas format
            error_data = {
                "answer": f"Error processing request: {str(e)}",
                "success": False,
                "error": str(e)
            }
            
            return WorkflowCanvasResponse(
                Code=1,
                Data=error_data,
                Status="Error",
                URL=workflow_request.url or "/",
                Method=workflow_request.Method or "POST",
                Headers=workflow_request.Headers or {},
                Body=error_data
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.request_count - self.success_count,
            "success_rate": f"{success_rate:.1f}%"
        }


# FastAPI routes for Workflow Canvas integration
def setup_workflow_canvas_routes(app: FastAPI, wfc_system: WFCIntegrationSystem):
    """Add Workflow Canvas specific routes to FastAPI app"""
    
    integration = WorkflowCanvasIntegration(wfc_system)
    
    @app.post("/workflow/execute")
    async def execute_workflow(request: WorkflowCanvasRequest) -> WorkflowCanvasResponse:
        """
        Main endpoint for Workflow Canvas RESTful.1 integration
        
        This endpoint matches the format expected by Workflow Canvas HTTP Server block:
        - Receives: url, Body, Params, Headers, Method
        - Returns: Code, Data, Status, Server, URL, Method, Headers, Body, Timeout
        
        Example request body:
        {
            "Body": {
                "query": "What machines have high vibration?",
                "mode": "enhanced",
                "response_style": "balanced"
            }
        }
        """
        return await integration.process_workflow_request(request)
    
    @app.post("/workflow/query")
    async def workflow_query(request: Request) -> WorkflowCanvasResponse:
        """
        Simplified endpoint - automatically wraps simple query in Workflow Canvas format
        
        Example request body:
        {
            "query": "List all machines in zone Z1",
            "mode": "graph"
        }
        """
        data = await request.json()
        
        # Wrap in Workflow Canvas format
        wfc_request = WorkflowCanvasRequest(
            url="/workflow/query",
            Body=data,
            Method="POST"
        )
        
        return await integration.process_workflow_request(wfc_request)
    
    @app.get("/workflow/stats")
    async def workflow_stats() -> Dict[str, Any]:
        """Get Workflow Canvas integration statistics"""
        return integration.get_stats()
    
    logger.info("✅ Workflow Canvas routes registered")
    logger.info("   - POST /workflow/execute (RESTful.1 format)")
    logger.info("   - POST /workflow/query (simplified format)")
    logger.info("   - GET /workflow/stats (statistics)")
