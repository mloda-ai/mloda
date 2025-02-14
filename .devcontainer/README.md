# Dev Containers Setup

This setup allows you to develop and test your code inside a Docker container. It uses Docker Compose to manage multiple containers and ensures that your development environment is consistent across different machines. The containers are configured to open in the `/workspace` directory, where your project files are located.

## How to Attach to a Dev Container

1. Press `Ctrl+Shift+P` to open the command palette.
2. Select `Remote-Containers: Attach to Running Container...`.
3. Choose the desired container from the list.

## How to Build and Start the Dev Containers

```sh
cd .devcontainer && docker compose up --build -d
```

## How to Stop and Remove the Dev Containers

```sh
docker compose down -v
```

## How to Stop the Dev Containers Without Removing Volumes

```sh
docker compose down
```

## How to Prune Docker Volumes

```sh
docker volume prune -f
