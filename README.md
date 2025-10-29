# WFC AI Integration System

## Overview

🏭 **AI-powered system for shopfloor data analysis, combining Retrieval-Augmented Generation (RAG) with graph database queries through a multi-agent architecture. Features an interactive query interface for real-time data exploration.**

## Features

- **Multi-Agent Architecture**: LLM-driven main agent orchestrates requests and delegates to:
  - **RAG Agent**: Semantic search and document retrieval from CSV data
  - **Graph Agent**: Structured queries to a Neo4j shopfloor graph database
  - **OpenAI Integration**: (Optional) Enhanced LLM responses
- **RAG System**: Converts CSVs to documents, builds vector embeddings (sentence-transformers), and enables semantic search (FAISS)
- **Graph Database**: Uses Neo4j and a contract-driven tool for safe, parameterized Cypher queries
- **Interactive Query Interface**: Real-time, command-driven interface for users
- **Async, extensible, and testable Python codebase**

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
- Copy `.env.template` to `.env` and fill in your credentials (especially OpenAI and Neo4j)

### 3. Run the System
```bash
# Interactive query interface (recommended)
python query_interface.py

# Or launch the orchestrator directly
python ai_system.py
```

## Query Modes
- `<query>`: Auto-detects best agent (RAG or Graph)
- `rag <query>`: Force semantic search (RAG)
- `graph <query>`: Force database query (Graph)
- `enhanced <query>`: Use OpenAI for LLM-powered responses (if configured)
- `simple|technical|executive <query>`: OpenAI with different response styles

## Main Components

- `ai_system.py`: Main orchestrator, demo, and interactive mode
- `query_interface.py`: Interactive CLI for user queries
- `ai_agent.py`: LLM-driven main agent (decision making, agent registration)
- `ai_brain.py`: OpenAI integration and enhanced RAG agent
- `ai_coordinator.py`: ACP (AI Control Protocol) for tool selection
- `graph_db.py`: Graph agent, natural language to Cypher mapping
- `rag_processor.py`: RAG system, document processing, vector search
- `mcp_tool.py`: ShopfloorGraphTool, contract-driven Neo4j access
- `mcp_tool_contract.json`: Defines all available graph operations
- `data/`: CSVs for nodes and relationships (shopfloor data)
- `schema/`: Graph schema documentation

## Architecture

```
┌──────────────┐   ┌──────────────┐
│  User Query  │─▶ │ Main Agent   │
└──────────────┘   └─────┬────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   ┌──────────────┐               ┌──────────────┐
   │  RAG Agent   │               │ Graph Agent  │
   └─────┬────────┘               └─────┬────────┘
         ▼                              ▼
   ┌──────────────┐               ┌──────────────┐
   │ Vector Store │               │ Neo4j DB     │
   └──────────────┘               └──────────────┘
```

## Example Commands

- `What machines have high vibration?`
- `rag Find maintenance documentation`
- `graph List sensors in zone Z1`
- `enhanced What's wrong with machine M100?`
- `simple Explain high vibration in simple terms`
- `technical Analyze sensor data patterns in detail`
- `executive Summarize facility operational status`

## Graph Operations (see `mcp_tool_contract.json`)
- `highVibrationMachines(threshold, unit)`
- `overdueWorkOrders()`
- `currentOperator(machineId)`
- `sensorsByZone(zoneId)`
- `dueForMaintenance(cutoff)`
- ...and more (see contract for full list)

## Data & Schema
- `data/`: CSVs for machines, sensors, operators, work orders, maintenance logs, facility zones, production batches, and relationships
- `schema/shopfloor_graph_schema.md`: Full LPG schema for the graph database

## Configuration

- `.env`: Environment variables (OpenAI, Neo4j, etc.)
- `requirements.txt`: Python dependencies

## Troubleshooting
- **Neo4j Connection**: Check credentials and network
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Data Files**: Ensure all CSVs are present in `data/`
- **Python Version**: Python 3.8+
- **RAG Model Download**: If behind a proxy, set SSL env vars as needed

## Advanced
- Extend by adding new agents or graph operations (see `ai_agent.py`, `graph_db.py`)
- All agents are async and can be registered with the main agent
- OpenAI integration is optional but recommended for enhanced responses

## License
MIT License - see LICENSE file for details

---

*Built with ❤️ for efficient shopfloor AI integration*