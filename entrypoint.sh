#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the path to the repository
REPO_PATH="/articutool_ws/src/articutool_ros2"

# Check if the repository directory and .git folder exist
if [ -d "$REPO_PATH" ] && [ -d "$REPO_PATH/.git" ]; then
  echo "Updating articutool_ros2 repository in $REPO_PATH..."
  cd "$REPO_PATH"
  git pull
  # Note: If 'git pull' brings changes that require rebuilding your ROS workspace
  # (e.g., changes to CMakeLists.txt, new packages), you would need to
  # manually run 'colcon build' inside the container or trigger a rebuild.
  # This script does not automatically rebuild the workspace after pulling.
  cd "/articutool_ws" # Return to the working directory
else
  echo "Warning: Repository at $REPO_PATH not found or is not a git repository. Skipping git pull."
  # Optionally, you could clone it here if it doesn't exist, though your Dockerfile already handles the initial clone.
  # Example:
  # if [ ! -d "$REPO_PATH" ]; then
  #   echo "Cloning articutool_ros2 repository..."
  #   mkdir -p /articutool_ws/src
  #   git clone https://github.com/personalrobotics/articutool_ros2.git "$REPO_PATH"
  # fi
fi

# Source ROS and Articutool workspace setup files
# This ensures the environment is set up correctly for the command that will be executed.
echo "Sourcing ROS and Articutool setup files..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source "/opt/ros/humble/setup.bash"
else
  echo "Warning: /opt/ros/humble/setup.bash not found."
fi

if [ -f "/articutool_ws/install/setup.bash" ]; then
  source "/articutool_ws/install/setup.bash"
else
  echo "Warning: /articutool_ws/install/setup.bash not found. This might be normal if the workspace hasn't been built yet during image creation."
fi

# Execute the command passed to the entrypoint
# (this will be the Dockerfile's CMD or arguments to 'docker run')
echo "Executing command: $@"
exec "$@"
