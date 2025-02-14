# Use the same VS Code Dev Container base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Set up a non-root user (matches `remoteUser: vscode` in devcontainer.json)
USER root
WORKDIR /workspace

# Install Docker-in-Docker feature
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | tee /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io \
    && rm -rf /var/lib/apt/lists/*

USER vscode

# Copy project files initially into the container
COPY ../ /workspace

# Set correct permissions
RUN sudo chown -R vscode:vscode /workspace

# Add .bashrc to change directory to /workspace
RUN echo 'cd /workspace' >> /home/vscode/.bashrc

# Install dependencies
RUN pip install --no-cache-dir -r /workspace/docs/requirements.txt -r /workspace/tests/requirements-test.txt
RUN pip install --no-cache-dir tox

# run tox -> container starts with tox env already installed
RUN cd /workspace && ~/.local/bin/tox || tox || echo "Tox failed but continuing..."

WORKDIR /workspace 

# Default shell
CMD ["bash", "-c", "cd /workspace && exec bash"]