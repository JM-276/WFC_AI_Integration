# WFC AI Integration - Industrial Edge App Configuration

This directory contains configuration files needed to deploy the WFC AI Integration as an Industrial Edge application.

## Files

- `app-manifest.json`: Industrial Edge Management app manifest
- `deployment-guide.md`: Step-by-step deployment instructions

## Quick Reference

### Required Environment Variables
- `NEO4J_URI`: Connection string to Neo4j database
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

### Optional Environment Variables
- `OPENAI_API_KEY`: For enhanced AI responses
- `OPENAI_MODEL`: OpenAI model selection (default: gpt-4)
- `USE_ENHANCED_MODE`: Enable/disable OpenAI integration
- `LOG_LEVEL`: Application logging level

### Ports
- **8000**: FastAPI REST API

### Resource Requirements
- **Memory**: 1-2 GB
- **CPU**: 0.5-2 cores
- **Storage**: Minimal (cache and data volumes)

## API Endpoints

Once deployed, the application exposes:

- `GET /`: Health check and API information
- `POST /query`: Process queries through multi-agent system
- `POST /enhanced-query`: AI-enhanced query processing
- `GET /status`: System status
- `GET /stats`: System statistics

## Integration with Workflow Canvas

After importing this app into Industrial Edge Management, it will appear as a node in Workflow Canvas under your custom IE Apps.

The node will expose:
- Input: Query text
- Output: Structured JSON response with analysis results
- Configuration: Environment variables for Neo4j and OpenAI
