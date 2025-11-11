# GitHub Setup Guide

Quick reference for setting up your GitHub repository for automated Docker builds.

## Step 1: Add GitHub Secrets

Your GitHub Actions workflow needs credentials to push images to Docker Hub.

1. Go to your repository: `https://github.com/lalukeland/WFC_AI_Integration`
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**

Add these two secrets:

### DOCKER_USERNAME
- **Name**: `DOCKER_USERNAME`
- **Value**: Your Docker Hub username (e.g., `lalukeland`)

### DOCKER_PASSWORD
- **Name**: `DOCKER_PASSWORD`
- **Value**: Your Docker Hub password or access token

**Security Tip**: Use a Docker Hub access token instead of your password:
- Go to https://hub.docker.com/settings/security
- Click **New Access Token**
- Give it a name (e.g., "GitHub Actions")
- Copy the token and use it as `DOCKER_PASSWORD`

## Step 2: Verify Workflow File

The workflow file is already created at:
```
.github/workflows/build-and-push.yml
```

This workflow automatically:
- Builds your Docker image
- Tags it with version numbers
- Pushes to Docker Hub
- Runs security scans

## Step 3: Trigger First Build

### Option A: Push to Branch
```bash
git add .
git commit -m "Add Docker and IE deployment files"
git push origin Luke
```

### Option B: Create Version Tag
```bash
git tag v1.0.0
git push origin v1.0.0
```

### Option C: Manual Trigger
1. Go to **Actions** tab in GitHub
2. Select "Build and Push Docker Image"
3. Click **Run workflow**
4. Choose branch (e.g., `Luke`)
5. Click **Run workflow**

## Step 4: Monitor Build

1. Go to **Actions** tab
2. Click on the running workflow
3. Watch the build progress
4. Verify all steps complete successfully
5. Check for the image in Docker Hub: `https://hub.docker.com/r/lalukeland/wfc-ai-integration`

## Expected Image Tags

After successful build, you'll have these tags:

- `latest` - Latest build from main branch
- `Luke` - Latest build from Luke branch
- `v1.0.0` - Specific version tags
- `Luke-<git-sha>` - Branch + commit SHA

## Troubleshooting

### Build Fails: "denied: requested access to the resource is denied"
- Check DOCKER_USERNAME and DOCKER_PASSWORD secrets
- Verify Docker Hub account is active
- Ensure you have push permissions to the repository

### Build Fails: "manifest for ... not found"
- This is normal on first build
- The workflow creates the repository automatically

### Image Not Appearing in Docker Hub
- Check workflow completed successfully (green checkmark)
- Verify secrets are named exactly: `DOCKER_USERNAME` and `DOCKER_PASSWORD`
- Check Docker Hub for new repository under your account

## Next Steps

Once the image is built and pushed:

1. ✅ Verify image exists in Docker Hub
2. ✅ Follow `industrial-edge/DEPLOYMENT_GUIDE.md` to import into IEM
3. ✅ Deploy to your Industrial Edge device
4. ✅ Use in Workflow Canvas!

## Updating Your App

Every time you push changes:

```bash
# Make your code changes
git add .
git commit -m "Your changes"
git push origin Luke

# For production releases
git tag v1.1.0
git push origin v1.1.0
```

GitHub Actions automatically rebuilds and pushes the new version!

## Questions?

- Check build logs in Actions tab
- Review `industrial-edge/DEPLOYMENT_GUIDE.md`
- Check Docker Hub for image status
