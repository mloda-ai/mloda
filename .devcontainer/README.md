# Dev Containers Setup

This setup allows you to develop and test your code inside a Docker container. It uses Docker Compose to manage multiple containers and ensures that your development environment is consistent across different machines. The containers are configured to open in the `/workspace` directory, where your project files are located.

## Available Containers

- **ai-agent-1**: Port 8000
- **ai-agent-2**: Port 8001
- **ai-agent-3**: Port 8002
- **ai-agent-4**: Port 8003

## How to Attach to a Dev Container via vscode

1. Press `Ctrl+Shift+P` to open the command palette.
2. Select `Remote-Containers: Attach to Running Container...`.
3. Choose the desired container from the list.

## How to Attach to a Dev Container via terminal

- **docker exec -it ai-agent-1 /bin/bash**
- **docker exec -it ai-agent-2 /bin/bash**
- **docker exec -it ai-agent-3 /bin/bash**
- **docker exec -it ai-agent-4 /bin/bash**

## Building and Starting Containers

### Build and Start ALL Containers (4 containers)

```sh
cd .devcontainer && docker compose --profile build-only build && docker compose up -d
```

Or use the shorthand (builds base image automatically):
```sh
cd .devcontainer && docker compose up --build -d
```

### Build and Start a SINGLE Container

```sh
# For ai-agent-1 (builds base image if needed)
cd .devcontainer && docker compose up --build -d ai-agent-1

# For ai-agent-2
cd .devcontainer && docker compose up --build -d ai-agent-2

# For ai-agent-3
cd .devcontainer && docker compose up --build -d ai-agent-3

# For ai-agent-4
cd .devcontainer && docker compose up --build -d ai-agent-4
```

### Build and Start Multiple Specific Containers

```sh
# Example: Start only ai-agent-1 and ai-agent-3
cd .devcontainer && docker compose up --build -d ai-agent-1 ai-agent-3
```

### Rebuild Base Image Only

```sh
# When you need to update the base image
cd .devcontainer && docker compose build mloda-base
```

## Stopping Containers

### Stop ALL Containers and Remove Volumes

```sh
docker compose down -v
```

### Stop ALL Containers Without Removing Volumes

```sh
docker compose down
```

### Stop a SINGLE Container

```sh
# For ai-agent-1
docker compose stop ai-agent-1

# For ai-agent-2
docker compose stop ai-agent-2

# For ai-agent-3
docker compose stop ai-agent-3

# For ai-agent-4
docker compose stop ai-agent-4
```

### Remove a SINGLE Container (keeping volume)

```sh
# Example for ai-agent-1
docker compose rm -f ai-agent-1
```

### Remove a SINGLE Container AND its Volume

```sh
# Example for ai-agent-1
docker compose rm -f ai-agent-1
docker volume rm .devcontainer_ai_agent_1_data
```

## Managing Docker Volumes

### List All Project Volumes

```sh
docker volume ls | grep devcontainer
```

### Prune All Unused Docker Volumes

```sh
docker volume prune -f
