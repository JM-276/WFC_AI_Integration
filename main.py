from fastapi import FastAPI, Request, HTTPException
from typing import Any, Dict, Optional
from pydantic import BaseModel
import logging

# Import your workflow system
from ai_system import WFCIntegrationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="WFC AI Integration API",
    description="API for WFC AI Multi-Agent System with RAG and Graph capabilities",
    version="1.0.0"
)

# Global system instance
wfc_system = None

# Request models
class QueryRequest(BaseModel):
    query: str
    preferred_agent: Optional[str] = None
    max_results: int = 5
    use_llm: bool = True
    response_mode: str = "balanced"

class EnhancedQueryRequest(BaseModel):
    query: str
    use_llm: bool = True
    response_mode: str = "balanced"
    max_results: int = 5

@app.on_event("startup")
async def startup_event():
    """Initialize the WFC system on startup"""
    global wfc_system
    logger.info("🚀 Starting WFC AI Integration API...")
    wfc_system = WFCIntegrationSystem(use_enhanced_mode=True)
    success = await wfc_system.initialize()
    if success:
        logger.info("✅ WFC System initialized successfully")
    else:
        logger.error("❌ Failed to initialize WFC System")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global wfc_system
    if wfc_system:
        await wfc_system.cleanup()
        logger.info("✅ WFC System cleanup complete")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "WFC AI Integration API is running",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Process a query through the multi-agent system",
            "POST /enhanced-query": "Process a query with AI-enhanced responses",
            "GET /status": "Get system status",
            "GET /stats": "Get system statistics"
        }
    }

@app.post("/query")
async def process_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a query through the WFC multi-agent system
    
    - **query**: The user's question or request
    - **preferred_agent**: Optional agent to use (rag, graph, enhanced)
    - **max_results**: Maximum number of results to return
    - **use_llm**: Whether to use LLM for enhanced responses
    - **response_mode**: Response style (simple, balanced, technical)
    """
    global wfc_system
    
    if not wfc_system or not wfc_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await wfc_system.process_query(
            query=request.query,
            preferred_agent=request.preferred_agent,
            max_results=request.max_results,
            use_llm=request.use_llm,
            response_mode=request.response_mode
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced-query")
async def process_enhanced_query(request: EnhancedQueryRequest) -> Dict[str, Any]:
    """
    Process a query with AI-enhanced capabilities (OpenAI integration)
    
    - **query**: The user's question or request
    - **use_llm**: Whether to use LLM for responses
    - **response_mode**: Response style (simple, balanced, technical)
    - **max_results**: Maximum number of results
    """
    global wfc_system
    
    if not wfc_system or not wfc_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await wfc_system.process_enhanced_query(
            query=request.query,
            use_llm=request.use_llm,
            response_mode=request.response_mode,
            max_results=request.max_results
        )
        return result
    except Exception as e:
        logger.error(f"Error processing enhanced query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the current status of the WFC system"""
    global wfc_system
    
    if not wfc_system:
        return {"status": "not_initialized", "initialized": False}
    
    return {
        "status": "operational" if wfc_system.initialized else "not_initialized",
        "initialized": wfc_system.initialized,
        "enhanced_mode": wfc_system.use_enhanced_mode,
        "stats": wfc_system.get_quick_stats()
    }

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get detailed system statistics"""
    global wfc_system
    
    if not wfc_system or not wfc_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return wfc_system.get_quick_stats()

@app.post("/run-workflow")
async def run_workflow(request: Request):
    """Legacy endpoint - redirects to /query"""
    data = await request.json()
    query = data.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    return await process_query(QueryRequest(query=query))