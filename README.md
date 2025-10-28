# WFC AI Integration System
## Luke's RAG Implementation with Multi-Agent Architecture

🏭 **Complete AI-powered system for shopfloor data analysis combining RAG (Retrieval-Augmented Generation) with graph database queries through a multi-agent architecture. Features an interactive query interface for real-time data exploration.**

## 🌟 Features

### Multi-Agent Architecture
- **Main Agent**: Orchestrates requests and delegates to appropriate secondary agents
- **RAG Agent**: Handles semantic search and document retrieval  
- **Graph Agent**: Manages structured database queries
- **MCP Integration**: Model Context Protocol for tool communication

### RAG System (Luke's Implementation)
- **Document Processing**: Converts CSV data into searchable documents
- **Vector Embeddings**: Uses sentence transformers for semantic understanding
- **Semantic Search**: FAISS-powered similarity search
- **Context Generation**: Formatted output for LLM augmentation

### Graph Database Integration
- **Neo4j Integration**: Connects to existing shopfloor graph database
- **Predefined Operations**: 5 specialized queries for common tasks
- **Natural Language Mapping**: Converts queries to graph operations
- **Real-time Data**: Live operational insights

## 🚀 Quick Start

### 1. Setup System
```bash
# Install dependencies and validate system
python setup.py
```

### 2. Configure OpenAI (Optional)
```bash
# Edit .env file and add your OpenAI API key
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the System
```bash
# Interactive query interface (recommended)
python interactive_query.py

# Or launch integrated system directly
python integration_system.py
```

### 4. Query Modes
- **Interactive Mode**: Real-time query interface with full control
- **Auto-detect**: System automatically chooses best agent
- **Forced RAG**: Use `rag <query>` for semantic search
- **Forced Graph**: Use `graph <query>` for database queries
- **OpenAI Enhanced**: Use `enhanced <query>` for AI-powered responses

## 📋 System Components

### Core Files
- `integration_system.py` - Complete system orchestrator
- `main_agent.py` - Primary coordination agent
- `rag_agent.py` - RAG secondary agent
- `graph_agent.py` - Graph database secondary agent
- `rag_system.py` - RAG core implementation
- `rag_mcp_server.py` - MCP server for RAG tools
- `shopfloor_tool.py` - Graph database tool (existing)

### Configuration
- `requirements.txt` - Python dependencies
- `shopfloor_tool_contract.json` - Graph operations contract
- `.env` - Environment variables (create from .env template)
- `openai_integration.py` - OpenAI integration module

### Data Sources
- `data/nodes_*.csv` - Entity data (machines, sensors, operators, etc.)
- `data/rels_*.csv` - Relationship data (connections, assignments, etc.)

## 🔧 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Main Agent    │
└─────────────────┘    └─────────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌─────────────────┐       ┌─────────────────┐
           │   RAG Agent     │       │  Graph Agent    │
           └─────────┬───────┘       └─────────┬───────┘
                     │                         │
                     ▼                         ▼
           ┌─────────────────┐       ┌─────────────────┐
           │  RAG MCP Server │       │ Shopfloor Tool  │
           └─────────┬───────┘       └─────────┬───────┘
                     │                         │
                     ▼                         ▼
           ┌─────────────────┐       ┌─────────────────┐
           │   Vector Store  │       │   Neo4j Graph   │
           │  (FAISS Index)  │       │    Database     │
           └─────────────────┘       └─────────────────┘
```

## 🎯 Interactive Query Interface

The system provides a powerful interactive interface where you can input any questions and get real-time responses. Launch it with:

```bash
python interactive_query.py
```

### Available Commands
- `<query>` - Auto-detect best agent for your question
- `rag <query>` - Force semantic search (RAG agent)
- `graph <query>` - Force database query (Graph agent)
- `enhanced <query>` - Use OpenAI with balanced responses
- `simple <query>` - Use OpenAI with simple explanations
- `technical <query>` - Use OpenAI with technical details
- `executive <query>` - Use OpenAI with executive summaries
- `help` - Show all available commands
- `examples` - Show detailed query examples
- `status` - Display system status
- `stats` - Show usage statistics
- `quit` - Exit the interface

### Example Queries

**Auto-Detect Mode (just type your question):**
- "What machines have high vibration?"
- "Show current operators"
- "Find overdue work orders"
- "List sensors in zone Z1"

**Semantic Search (RAG):**
- "rag Find maintenance procedures for equipment"
- "rag What sensor monitoring documentation exists?"
- "rag Search production batch information"

**Database Queries (Graph):**
- "graph What machines have vibration above 7.5?"
- "graph Show operator for machine M100"
- "graph Which work orders are overdue?"

**OpenAI Enhanced (if configured):**
- "enhanced What's wrong with machine M100 and how do I fix it?"
- "simple Explain high vibration in simple terms"
- "technical Analyze sensor data patterns in detail"
- "executive Summarize facility operational status"

## 🛠️ Graph Database Operations

The system includes 5 predefined graph operations:

1. **highVibrationMachines(threshold, unit)** - Find machines with excessive vibration
2. **overdueWorkOrders()** - Get overdue maintenance work
3. **currentOperator(machineId)** - Find current machine operator  
4. **sensorsByZone(zoneId)** - List sensors in specific zone
5. **dueForMaintenance(cutoff)** - Find machines needing maintenance

Each operation includes:
- Description and input schema
- Parameterized Cypher query
- Output field definitions
- Example usage

## 🔐 Configuration

### Environment Variables
```bash
# Neo4j Database
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# RAG Configuration (optional)
RAG_MODEL_NAME=all-MiniLM-L6-v2
RAG_CACHE_DIR=./rag_cache
```

### Dependencies
- Python 3.8+
- Neo4j Python driver
- Sentence Transformers
- FAISS (CPU version)
- Pandas, NumPy, scikit-learn
- MCP (Model Context Protocol)

## 📊 System Statistics

The system tracks:
- Request processing metrics
- Agent usage patterns  
- Query execution times
- Success/failure rates
- Vector store performance
- Database connection health

## 🧪 Testing

### Individual Components
```bash
# Test RAG system
python rag_system.py "test query"

# Test RAG agent
python rag_agent.py

# Test graph agent  
python graph_agent.py

# Test integration
python integration_system.py
```

### Validation
```bash
# Run complete system validation
python setup.py
```

## 🔍 Troubleshooting

### Common Issues
1. **Neo4j Connection**: Verify credentials and network access
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Data Files**: Ensure all CSV files are present in `data/` directory
4. **Python Version**: Requires Python 3.8 or higher
5. **Memory**: RAG system requires sufficient RAM for embeddings

### Debug Mode
Set `LOG_LEVEL=DEBUG` for detailed logging

## 🚀 Advanced Usage

### Custom Agents
Extend the system by implementing new secondary agents:
```python
class CustomAgent:
    async def initialize(self):
        # Agent setup
        pass
    
    def get_status(self):
        # Return agent status
        pass
```

### MCP Tools
Add new tools to the RAG MCP server:
```python
@server.call_tool()
async def custom_tool(name: str, arguments: Dict[str, Any]):
    # Custom tool implementation
    pass
```

## 📈 Performance

- **RAG Queries**: ~0.1-0.5 seconds for semantic search
- **Graph Queries**: ~0.05-0.2 seconds for structured data
- **Hybrid Queries**: ~0.2-0.7 seconds combined
- **Vector Index**: Supports thousands of documents
- **Concurrent Requests**: Async-capable architecture

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

---

**Password for Neo4j Database**: `U32P3onr7idgSWbqklVReZQ8BVRH_BWH3_A5Oj83oq0`

*Built with ❤️ for efficient shopfloor AI integration*