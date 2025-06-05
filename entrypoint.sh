#!/bin/bash
set -e

ARTICUTOOL_REPO_PATH="/articutool_ws/src/articutool_ros2"
# Path to the root of the ada_ros2 checkout (where .git for sparse checkout is)
ADA_ROS2_CHECKOUT_PATH="/articutool_ws/src/ada_ros2_checkout_dir"
ADA_ROS2_REMOTE_BRANCH="jjaime2/articutool-integration"

update_repo() {
  local repo_path="$1"
  local repo_name="$2"
  local branch_to_pull="$3"

  if [ -d "$repo_path" ] && [ -d "$repo_path/.git" ]; then
    echo "Updating $repo_name repository in $repo_path..."
    cd "$repo_path"
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo "Current branch in $repo_name: $current_branch. Intended: $branch_to_pull"

    # Stash any local changes to avoid conflicts (optional, use with caution)
    # echo "Stashing local changes in $repo_name if any..."
    # git stash push -u -m "Entrypoint stash before pull on $(date)" || true

    echo "Attempting to fetch from origin for $repo_name..."
    git fetch origin "$branch_to_pull" || echo "Warning: Fetch failed for $repo_name branch $branch_to_pull"

    echo "Attempting to pull (merge) $branch_to_pull for $repo_name..."
    if git pull origin "$branch_to_pull"; then # Or use git rebase or git reset --hard
      echo "$repo_name updated successfully from $branch_to_pull."
    else
      echo "Error: 'git pull origin $branch_to_pull' for $repo_name failed. Check for conflicts or ensure branch exists and tracks remote."
      # Attempt to pop stash if stashed (optional)
      # echo "Attempting to pop stash for $repo_name..."
      # git stash pop || echo "Warning: Stash pop failed for $repo_name or no stash found."
    fi
    cd "/articutool_ws"
  else
    echo "Warning: Repository $repo_name at $repo_path not found or is not a git repository. Skipping git pull."
  fi
}

# Update articutool_ros2
update_repo "$ARTICUTOOL_REPO_PATH" "articutool_ros2" # Pulls its currently tracked branch

# Update the ada_ros2 repository (which contains ada_moveit and ada_description via sparse checkout)
update_repo "$ADA_ROS2_CHECKOUT_PATH" "ada_ros2 (for sparse content)" "$ADA_ROS2_REMOTE_BRANCH"

echo "Changing to workspace directory: /articutool_ws"
cd "/articutool_ws"

echo "Sourcing ROS base setup file..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source "/opt/ros/humble/setup.bash"
else
  echo "Error: /opt/ros/humble/setup.bash not found." >&2
  exit 1
fi

echo "Building the Articutool workspace with colcon..."
colcon build --symlink-install

echo "Sourcing the local Articutool workspace setup file..."
if [ -f "/articutool_ws/install/setup.bash" ]; then
  source "/articutool_ws/install/setup.bash"
else
  echo "Error: /articutool_ws/install/setup.bash not found after colcon build." >&2
  exit 1
fi

echo "Entrypoint CMD: $@"
if [ $# -gt 0 ]; then
  echo "Executing provided command: $@"
  exec "$@"
else
  echo "Executing default command: ros2 launch articutool_system articutool.launch.py sim:=real"
  exec ros2 launch articutool_system articutool.launch.py sim:=real
fi
