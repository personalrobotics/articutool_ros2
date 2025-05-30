#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the path to the repository
REPO_PATH="/articutool_ws/src/articutool_ros2"

# Check if the repository directory and .git folder exist
if [ -d "$REPO_PATH" ] && [ -d "$REPO_PATH/.git" ]; then
  echo "Updating articutool_ros2 repository in $REPO_PATH..."
  cd "$REPO_PATH"
  git pull
  # The script will 'cd "/articutool_ws"' later, before colcon build
else
  echo "Warning: Repository at $REPO_PATH not found or is not a git repository. Skipping git pull."
  # Optionally, you could clone it here if it doesn't exist.
  # However, the Dockerfile should generally handle the initial clone.
fi

# Ensure we are in the workspace root for subsequent commands
echo "Changing to workspace directory: /articutool_ws"
cd "/articutool_ws"

# Source base ROS environment first
echo "Sourcing ROS base setup file..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source "/opt/ros/humble/setup.bash"
else
  echo "Error: /opt/ros/humble/setup.bash not found. ROS environment cannot be set up."
  exit 1
fi

# Now, build the workspace after potentially pulling updates
echo "Building the Articutool workspace with colcon..."
colcon build --symlink-install

# Source the local (now freshly built) workspace setup file
echo "Sourcing the local Articutool workspace setup file..."
if [ -f "/articutool_ws/install/setup.bash" ]; then
  source "/articutool_ws/install/setup.bash"
else
  echo "Error: /articutool_ws/install/setup.bash not found after colcon build. Build might have failed or the path is incorrect."
  exit 1
fi

# Execute the command passed to the entrypoint (CMD in Dockerfile or docker run arguments).
# If no command is passed, run the default articutool launch command.
if [ $# -gt 0 ]; then
  echo "Executing provided command: $@"
  exec "$@"
else
  echo "Executing default command: ros2 launch articutool_system articutool.launch.py sim:=real"
  exec ros2 launch articutool_system articutool.launch.py sim:=real
fi
