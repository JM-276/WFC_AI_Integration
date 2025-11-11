# Industrial Edge Deployment Guide

## Complete Process: From GitHub to Workflow Canvas

This guide walks you through deploying the WFC AI Integration application as a node in Siemens Workflow Canvas.

---

## Prerequisites

### 1. Required Accounts
- GitHub account with access to `lalukeland/WFC_AI_Integration`
- Docker Hub account (or Azure/AWS Container Registry)
- Siemens Industrial Edge Management access
- Neo4j database (can be cloud-hosted or on-premise)

### 2. Required Secrets

Set up these secrets in your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password or access token

---

## Phase 1: Build & Publish Container Image

### Step 1: Configure GitHub Secrets

1. Go to your repository: `github.com/lalukeland/WFC_AI_Integration`
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Add the following secrets:
   ```
   DOCKER_USERNAME=your-dockerhub-username
   DOCKER_PASSWORD=your-dockerhub-token
   ```

### Step 2: Trigger Build

The GitHub workflow automatically builds on:
- Push to `main` or `Luke` branches
- Creating version tags (e.g., `v1.0.0`)
- Manual trigger via "Actions" tab

**Manual Trigger:**
1. Go to **Actions** tab
2. Select "Build and Push Docker Image"
3. Click **Run workflow**
4. Select branch (e.g., `Luke`)
5. Click **Run workflow**

### Step 3: Verify Image Published

After the workflow completes (~5-10 minutes):

1. Check Docker Hub: `https://hub.docker.com/r/lalukeland/wfc-ai-integration`
2. Verify tags are present:
   - `latest` (from main branch)
   - `Luke` (from Luke branch)
   - Version tags (e.g., `v1.0.0`)

---

## Phase 2: Import to Industrial Edge Management

### Step 1: Access IEM Console

1. Log in to Siemens Industrial Edge Management
2. Navigate to **Applications** → **My Applications**

### Step 2: Create New Application

1. Click **Add Application**
2. Select **Import from Container Registry**
3. Fill in details:
   - **Name**: WFC AI Integration
   - **Description**: AI-powered shopfloor analysis with RAG and graph capabilities
   - **Registry Type**: Docker Hub
   - **Image URL**: `docker.io/lalukeland/wfc-ai-integration:latest`
   - **Pull Credentials**: (if repository is private)

### Step 3: Configure Application

#### Environment Variables
Configure these in IEM:

**Required:**
- `NEO4J_URI`: `bolt://your-neo4j-server:7687`
- `NEO4J_USER`: `neo4j`
- `NEO4J_PASSWORD`: Your Neo4j password

**Optional:**
- `OPENAI_API_KEY`: Your OpenAI key (for enhanced mode)
- `OPENAI_MODEL`: `gpt-4` or `gpt-3.5-turbo`
- `USE_ENHANCED_MODE`: `true` or `false`
- `LOG_LEVEL`: `INFO`, `DEBUG`, or `ERROR`

#### Ports
- **Port 8000**: HTTP API endpoint

#### Resources
- **Memory**: 1-2 GB
- **CPU**: 0.5-2 cores

### Step 4: Publish Application

1. Review all settings
2. Click **Save & Publish**
3. Wait for validation to complete
4. Application status should show "Published"

---

## Phase 3: Deploy to Edge Device

### Step 1: Assign to Edge Device

1. In IEM, go to **Edge Devices**
2. Select your target device
3. Navigate to **Applications** tab
4. Click **Install Application**
5. Select **WFC AI Integration**
6. Configure device-specific settings if needed
7. Click **Install**

### Step 2: Verify Installation

1. Wait for installation to complete (~2-5 minutes)
2. Check application status: should show "Running"
3. View logs to verify startup:
   ```
   🚀 Starting WFC AI Integration API...
   ✅ WFC System initialized successfully
   ```

### Step 3: Test API Connectivity

From a terminal on the Edge device or network:

```bash
# Health check
curl http://<edge-device-ip>:8000/

# Check status
curl http://<edge-device-ip>:8000/status

# Test query
curl -X POST http://<edge-device-ip>:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What machines need maintenance?"}'
```

---

## Phase 4: Use in Workflow Canvas

### Step 1: Open Workflow Canvas

1. Access your Industrial Edge device
2. Open **Workflow Canvas** application
3. Create a new workflow or open existing

### Step 2: Find WFC AI Integration Node

1. In the node palette (left sidebar)
2. Expand **Industrial Edge Apps** category
3. Look for **WFC AI Integration** node
4. It should display with the icon and name you configured

### Step 3: Drag and Configure Node

1. **Drag** the WFC AI Integration node onto the canvas
2. **Double-click** to configure:
   - **Node Name**: (optional, e.g., "AI Query Processor")
   - **Query Input**: Wire from input node or set static query
   - **Response Mode**: `simple`, `balanced`, or `technical`
   - **Max Results**: Number of results to return

### Step 4: Connect to Workflow

Example workflow:

```
[Inject Node] → [WFC AI Integration] → [Debug/Dashboard]
     ↓
  "What machines
   have high 
   vibration?"
```

**Input Structure:**
```json
{
  "query": "List all machines in zone Z1",
  "max_results": 5,
  "use_llm": true,
  "response_mode": "balanced"
}
```

**Output Structure:**
```json
{
  "response": "Found 3 machines...",
  "data": [...],
  "agent_used": "graph",
  "execution_time": 0.45
}
```

### Step 5: Deploy and Test

1. Click **Deploy** button (top right)
2. Trigger the workflow
3. Check debug output for results
4. Verify correct data is returned

---

## Troubleshooting

### Build Fails on GitHub Actions

**Problem**: Workflow fails during build
**Solution**:
- Check secrets are correctly set
- Verify Dockerfile syntax
- Review build logs in Actions tab

### Image Not Found in IEM

**Problem**: Cannot pull image from registry
**Solution**:
- Verify image URL is correct
- Check registry credentials if repository is private
- Ensure image was successfully pushed (check Docker Hub)

### Application Won't Start

**Problem**: Container exits immediately
**Solution**:
- Check environment variables are set correctly
- Verify Neo4j connection (URI, username, password)
- Review application logs in IEM

### Neo4j Connection Failed

**Problem**: App can't connect to Neo4j
**Solution**:
- Verify `NEO4J_URI` format: `bolt://hostname:7687`
- Check network connectivity from Edge device to Neo4j
- Confirm Neo4j credentials are correct
- Ensure Neo4j is running and accessible

### OpenAI Enhanced Mode Not Working

**Problem**: Enhanced queries fail
**Solution**:
- Verify `OPENAI_API_KEY` is set correctly
- Check API key has credits/quota
- Set `USE_ENHANCED_MODE=false` to disable and use basic mode

---

## Updating the Application

### Release New Version

1. Make code changes in GitHub
2. Create a new tag:
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
3. GitHub Actions automatically builds and pushes new version
4. In IEM, update application to new image tag
5. Redeploy to Edge devices

### Rolling Update

1. Update image tag in IEM to new version
2. Click **Update** on deployed instances
3. IEM performs rolling update with zero downtime

---

## Best Practices

### Version Management
- Use semantic versioning (v1.0.0, v1.1.0, etc.)
- Tag production releases
- Use branch names for development (e.g., `Luke`, `dev`)

### Security
- Never commit secrets to Git
- Use GitHub Secrets for credentials
- Rotate API keys regularly
- Use private container registry for production

### Monitoring
- Enable IEM application logging
- Set up alerts for container failures
- Monitor resource usage (CPU, memory)
- Track API response times

### Performance
- Adjust memory limits based on workload
- Use caching for RAG embeddings
- Consider external Neo4j for scale
- Monitor query execution times

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Repository                         │
│  (Source Code + Dockerfile + GitHub Actions)                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Automated CI/CD)
┌─────────────────────────────────────────────────────────────┐
│              Container Registry (Docker Hub)                 │
│         docker.io/lalukeland/wfc-ai-integration             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Pull Image)
┌─────────────────────────────────────────────────────────────┐
│         Siemens Industrial Edge Management (IEM)            │
│              (App Configuration & Publishing)               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Deploy)
┌─────────────────────────────────────────────────────────────┐
│              Industrial Edge Device                          │
│  ┌─────────────────────────────────────────────────┐       │
│  │     Docker Container: WFC AI Integration        │       │
│  │  (FastAPI + RAG + Graph + Multi-Agent System)   │       │
│  └─────────────────────────────────────────────────┘       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (Available as Node)
┌─────────────────────────────────────────────────────────────┐
│              Workflow Canvas                                 │
│  [Input] → [WFC AI Integration] → [Output/Dashboard]       │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. ✅ Set up GitHub secrets
2. ✅ Trigger first build
3. ✅ Verify image in Docker Hub
4. ✅ Import to IEM
5. ✅ Deploy to Edge device
6. ✅ Test in Workflow Canvas
7. ✅ Build your first AI-powered workflow!

---

## Support & Resources

- **GitHub Issues**: `github.com/lalukeland/WFC_AI_Integration/issues`
- **Docker Hub**: `hub.docker.com/r/lalukeland/wfc-ai-integration`
- **Siemens IEM Docs**: Industrial Edge Management documentation
- **API Reference**: See `industrial-edge/README.md`

---

*Last updated: November 2025*
