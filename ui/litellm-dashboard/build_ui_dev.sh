#!/bin/bash

# Build script for development - outputs to custom directory
# This won't touch the original litellm/proxy/_experimental/out/

# Check if nvm is not installed
if ! command -v nvm &> /dev/null; then
  # Install nvm
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

  # Source nvm script in the current session
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

# Use nvm to set the required Node.js version
nvm use v20

# Check if nvm use was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to switch to Node.js v20. Deployment aborted."
  exit 1
fi

# print contents of ui_colors.json
echo "Contents of ui_colors.json:"
cat ui_colors.json

# Run npm build
npm run build

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Build successful. Copying files to custom directory..."

  # Specify YOUR custom destination directory (won't conflict with upstream)
  # Same structure as upstream (_experimental/out) but in your own folder
  destination_dir="../../litellm/proxy/_my_experimental/out"

  # Create destination directory if it doesn't exist
  mkdir -p "$destination_dir"

  # Remove existing files in the destination directory
  rm -rf "$destination_dir"/*

  # Copy the contents of the output directory to your custom destination
  cp -r ./out/* "$destination_dir"

  rm -rf ./out

  echo "✅ Deployment completed!"
  echo "📂 UI built to: litellm/proxy/_my_experimental/out/"
  echo ""
  echo "To use this UI with proxy, add to your config.yaml:"
  echo "  general_settings:"
  echo "    ui_access_mode: admin"
  echo "    ui_path: \"litellm/proxy/_my_experimental/out\""
else
  echo "Build failed. Deployment aborted."
fi
