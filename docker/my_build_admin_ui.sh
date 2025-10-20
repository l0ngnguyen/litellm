#!/bin/bash

# Custom Admin UI build script for Docker
# This script builds the UI to _my_experimental directory to avoid conflicts

# Don't exit on error - let it continue
# set -e

echo "=== Custom Admin UI Build for Docker ==="
echo

# print current dir
pwd

# For custom builds, we always build the UI
echo "Building Custom Admin UI..."

# Install dependencies
# Check if we are on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    # Install dependencies using Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew not found. Please install Homebrew and try again."
        exit 1
    fi
    brew update
    brew install curl
else
    # Assume Linux, try using apt-get
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y curl
    elif command -v apk &> /dev/null; then
        # Try using apk if apt-get is not available
        apk update
        apk add curl
    else
        echo "Error: Unsupported package manager. Cannot install dependencies."
        exit 1
    fi
fi

# Install nvm and Node.js v20
echo "Installing nvm and Node.js v20..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

# Export NVM environment
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install and use Node.js v20
nvm install 20
nvm use 20

echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"

# Create ui_colors.json if it doesn't exist (use default)
if [ ! -f "ui/litellm-dashboard/ui_colors.json" ]; then
    echo '{"primary": "#1E40AF", "secondary": "#9333EA"}' > ui/litellm-dashboard/ui_colors.json
    echo "Created default ui_colors.json"
fi

# Create destination directory
mkdir -p litellm/proxy/_my_experimental/out

# cd into /ui/litellm-dashboard
cd ui/litellm-dashboard

echo "Current directory: $(pwd)"
echo "Installing npm dependencies..."

# Install dependencies
npm install

echo "Building UI..."
# Build directly without using my_build_ui.sh to avoid nvm conflicts
npm run build

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Build successful. Copying files..."

  # Specify the custom destination directory
  destination_dir="../../litellm/proxy/_my_experimental/out"

  # Create destination directory if it doesn't exist
  mkdir -p "$destination_dir"

  # Remove existing files in the destination directory
  echo "Cleaning destination directory..."
  rm -rf "$destination_dir"/*

  # Copy the contents of the output directory to the specified destination
  echo "Copying build output to $destination_dir..."
  cp -r ./out/* "$destination_dir"

  # Clean up local build output
  rm -rf ./out

  echo "UI files copied to $destination_dir"
else
  echo "Build failed. Deployment aborted."
  exit 1
fi

# return to root directory
cd ../..

echo ""
echo "=== Custom Admin UI build completed ==="
