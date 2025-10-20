#!/bin/bash

# Custom UI build script - outputs to litellm/proxy/_my_experimental
# This avoids conflicts with upstream code in _experimental directory

set -e  # Exit on error

echo "=== Custom LiteLLM UI Build Script ==="
echo "Output directory: litellm/proxy/_my_experimental/out"
echo ""

# Check if nvm is not installed
if ! command -v nvm &> /dev/null; then
  echo "Installing nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

  # Source nvm script in the current session
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

# Use nvm to set the required Node.js version
echo "Switching to Node.js v20..."
nvm use v20

# Check if nvm use was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to switch to Node.js v20. Build aborted."
  exit 1
fi

# Print current directory and Node version
echo ""
echo "Current directory: $(pwd)"
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"
echo ""

# Print contents of ui_colors.json if it exists
if [ -f "ui_colors.json" ]; then
  echo "Contents of ui_colors.json:"
  cat ui_colors.json
  echo ""
fi

# Run npm build
echo "Running npm build..."
npm run build

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo ""
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

  echo ""
  echo "=== Build and deployment completed successfully ==="
  echo "UI built to: $destination_dir"
else
  echo ""
  echo "Build failed. Deployment aborted."
  exit 1
fi
