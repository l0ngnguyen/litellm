# Custom Build Scripts for LiteLLM

This directory contains custom build scripts with `my_*` prefix to avoid conflicts with upstream LiteLLM code.

## Overview

These scripts build LiteLLM with a custom UI that outputs to `litellm/proxy/_my_experimental/` instead of the default `_experimental/` directory. This allows you to maintain custom changes without conflicts when merging upstream updates.

## Files

- **`my_build_ui.sh`** - Builds the UI to `litellm/proxy/_my_experimental/out/`
- **`my_Dockerfile`** - Custom Dockerfile that uses `my_build_admin_ui.sh`
- **`my_build_and_push.sh`** - Builds and pushes Docker images with date-based tags
- **`docker/my_build_admin_ui.sh`** - Docker build helper script

## Quick Start

### 1. Build UI Only (Local Development)

```bash
cd ui/litellm-dashboard
./my_build_ui.sh
```

This will:
- Install Node.js v20 via nvm
- Build the UI
- Output to `../../litellm/proxy/_my_experimental/out/`

### 2. Build and Push Docker Image

```bash
# Set your GitHub username (optional - auto-detected from git config)
export GITHUB_USERNAME=your-github-username

# Create and set your GitHub Personal Access Token
# Create at: https://github.com/settings/tokens
# Required scopes: write:packages, read:packages, delete:packages
export CR_PAT=your_github_token

# Login to GitHub Container Registry
echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Build and push
./my_build_and_push.sh
```

This will:
- Build the Docker image using `my_Dockerfile`
- Tag with date-based versions (e.g., `2025-01-20`, `2025-01-20-1430`)
- Push to GitHub Container Registry
- Available at `ghcr.io/<username>/litellm:latest`

## Image Tags

The build script creates three tags:

1. **Date tag**: `YYYY-MM-DD` (e.g., `2025-01-20`)
2. **DateTime tag**: `YYYY-MM-DD-HHMM` (e.g., `2025-01-20-1430`)
3. **Latest tag**: `latest`

## Running the Custom Image

```bash
# Pull the image
docker pull ghcr.io/<username>/litellm:2025-01-20

# Run the container
docker run -p 4000:4000 ghcr.io/<username>/litellm:2025-01-20

# Or use latest
docker run -p 4000:4000 ghcr.io/<username>/litellm:latest
```

## Directory Structure

```
litellm/
├── my_Dockerfile                          # Custom Dockerfile
├── my_build_and_push.sh                   # Build and push script
├── docker/
│   └── my_build_admin_ui.sh              # Docker UI build helper
├── ui/
│   └── litellm-dashboard/
│       └── my_build_ui.sh                # UI build script
└── litellm/
    └── proxy/
        ├── _experimental/                 # Upstream UI (DO NOT MODIFY)
        └── _my_experimental/              # Custom UI (SAFE TO MODIFY)
            └── out/                       # Built UI files
```

## Why Custom Scripts?

These scripts prevent merge conflicts with upstream LiteLLM by:

1. **Separate output directory**: `_my_experimental/` instead of `_experimental/`
2. **Prefixed filenames**: All custom files use `my_*` prefix
3. **No upstream modifications**: Original build scripts remain unchanged

## Troubleshooting

### Build fails with "Node.js v20 not found"

The script will automatically install Node.js v20 via nvm. If it fails:

```bash
# Manually install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
source ~/.nvm/nvm.sh

# Install Node.js v20
nvm install v20
nvm use v20
```

### Docker push fails with "authentication required"

Make sure you're logged in to GitHub Container Registry:

```bash
# Create a token at: https://github.com/settings/tokens
export CR_PAT=your_token
echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

### Image not visible on GitHub

GitHub packages are private by default. To make it public:

1. Go to `https://github.com/<username>?tab=packages`
2. Click on your `litellm` package
3. Go to Package settings
4. Scroll to "Danger Zone"
5. Click "Change visibility" → "Public"

## Configuration

### Custom UI Colors

Edit `ui/litellm-dashboard/ui_colors.json` to customize the UI:

```json
{
  "primary": "#1E40AF",
  "secondary": "#9333EA"
}
```

### GitHub Registry

By default, images are pushed to `ghcr.io/<username>/litellm`. To use a different registry:

Edit `my_build_and_push.sh` and change:

```bash
REGISTRY="ghcr.io"
IMAGE_NAME="litellm"
```

## Development Workflow

1. Make changes to the UI in `ui/litellm-dashboard/`
2. Test locally: `cd ui/litellm-dashboard && ./my_build_ui.sh`
3. Verify at `http://localhost:4000` (if running proxy)
4. Build Docker image: `./my_build_and_push.sh`
5. Deploy the new image

## Support

For issues related to:
- **Custom build scripts**: Open an issue in your fork
- **LiteLLM core**: https://github.com/BerriAI/litellm/issues
- **Docker**: Check Docker logs with `docker logs <container_id>`

## License

These scripts are provided as-is for custom LiteLLM deployments. LiteLLM itself is licensed under the MIT License.
