# 🚀 Quick Start Checklist

Follow this checklist to deploy your WFC AI Integration to Workflow Canvas.

## ✅ Phase 1: GitHub Setup (5 minutes)

- [ ] 1. Go to GitHub repository settings
- [ ] 2. Add secret: `DOCKER_USERNAME` = your Docker Hub username
- [ ] 3. Add secret: `DOCKER_PASSWORD` = your Docker Hub token
- [ ] 4. Commit and push all new files to your repository
- [ ] 5. Go to Actions tab and trigger "Build and Push Docker Image"
- [ ] 6. Wait for build to complete (~5-10 minutes)
- [ ] 7. Verify image at: `hub.docker.com/r/lalukeland/wfc-ai-integration`

📖 **Detailed Guide**: See `GITHUB_SETUP.md`

---

## ✅ Phase 2: Industrial Edge Management (10 minutes)

- [ ] 1. Log in to Siemens IEM console
- [ ] 2. Navigate to Applications → My Applications
- [ ] 3. Click "Add Application" → "Import from Container Registry"
- [ ] 4. Enter image URL: `docker.io/lalukeland/wfc-ai-integration:latest`
- [ ] 5. Configure environment variables:
  - [ ] `NEO4J_URI` = your Neo4j connection string
  - [ ] `NEO4J_USER` = neo4j
  - [ ] `NEO4J_PASSWORD` = your password
  - [ ] `OPENAI_API_KEY` = (optional) your OpenAI key
- [ ] 6. Set port 8000 for HTTP API
- [ ] 7. Set memory: 1-2 GB, CPU: 0.5-2 cores
- [ ] 8. Click "Save & Publish"
- [ ] 9. Wait for validation to complete

📖 **Detailed Guide**: See `industrial-edge/DEPLOYMENT_GUIDE.md`

---

## ✅ Phase 3: Deploy to Edge Device (5 minutes)

- [ ] 1. In IEM, go to Edge Devices
- [ ] 2. Select your target device
- [ ] 3. Click Applications tab → "Install Application"
- [ ] 4. Select "WFC AI Integration"
- [ ] 5. Review settings and click "Install"
- [ ] 6. Wait for installation (~2-5 minutes)
- [ ] 7. Verify status shows "Running"
- [ ] 8. Test API:
  ```bash
  curl http://<edge-device-ip>:8000/status
  ```

---

## ✅ Phase 4: Use in Workflow Canvas (5 minutes)

- [ ] 1. Open Workflow Canvas on your Edge device
- [ ] 2. Look for "WFC AI Integration" in node palette
- [ ] 3. Drag node onto canvas
- [ ] 4. Configure node (double-click):
  - [ ] Set query or wire from input
  - [ ] Choose response mode
  - [ ] Set max results
- [ ] 5. Connect to workflow (input → WFC node → output)
- [ ] 6. Deploy workflow
- [ ] 7. Test with sample query:
  ```json
  {
    "query": "What machines need maintenance?",
    "max_results": 5
  }
  ```
- [ ] 8. Verify output in debug/dashboard

---

## 🎉 Success!

You now have an AI-powered node in Workflow Canvas!

### What You Can Do Now:

- ✅ Ask natural language questions about your shopfloor
- ✅ Get semantic search results from RAG system
- ✅ Query graph database with plain English
- ✅ Get AI-enhanced responses (if OpenAI configured)
- ✅ Integrate AI into your automation workflows

### Example Queries to Try:

- "What machines have high vibration?"
- "List all overdue work orders"
- "Which sensors are in zone Z1?"
- "Show machines due for maintenance"
- "What's the status of machine M100?"

---

## 🔄 Making Updates

When you make code changes:

```bash
# Commit changes
git add .
git commit -m "Your update description"
git push origin Luke

# Create version tag for production
git tag v1.1.0
git push origin v1.1.0
```

GitHub Actions rebuilds automatically. Then update in IEM and redeploy to Edge devices.

---

## 📚 Documentation

- **GitHub Setup**: `GITHUB_SETUP.md`
- **Full Deployment Guide**: `industrial-edge/DEPLOYMENT_GUIDE.md`
- **API Reference**: `industrial-edge/README.md`
- **Main README**: `README.md`

---

## 🆘 Need Help?

Check the troubleshooting sections in:
- `GITHUB_SETUP.md` - GitHub/Docker issues
- `industrial-edge/DEPLOYMENT_GUIDE.md` - IEM/deployment issues

---

**Time to Complete**: ~25-30 minutes total
**Difficulty**: Intermediate
**Prerequisites**: GitHub, Docker Hub, IEM access, Neo4j database

*Let's build something amazing! 🚀*
