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
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
       | tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io \
  && rm -rf /var/lib/apt/lists/*

USER vscode

RUN curl -fsSL https://claude.ai/install.sh | bash

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/vscode/.local/bin:$PATH"

# Copy project files with correct ownership
COPY --chown=vscode:vscode ./ /workspace

# Convenience: start in /workspace on shell open
RUN echo 'cd /workspace' >> /home/vscode/.bashrc

# Set tox work directory in user's home to avoid volume mount conflicts
ENV TOX_WORK_DIR=/home/vscode/.tox-cache
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/.tox-cache/python310

# Create tox cache directory with proper ownership
RUN mkdir -p $TOX_WORK_DIR && chown -R vscode:vscode $TOX_WORK_DIR

# Install tox with uv backend
RUN uv tool install tox --with tox-uv

# Pre-build the single shared tox env
RUN tox --workdir $TOX_WORK_DIR -e python310 --notest -vv || true

# Sync all extras
RUN uv sync --all-extras || true

# Auto-activate the shared tox env in interactive shells
RUN echo '[ -f /home/vscode/.tox-cache/python310/bin/activate ] && source /home/vscode/.tox-cache/python310/bin/activate' \
      >> /home/vscode/.bashrc

# Default shell
CMD ["bash", "-c", "cd /workspace && exec bash"]
