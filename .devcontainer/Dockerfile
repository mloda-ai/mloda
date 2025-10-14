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

# Consistent Python path for build/run
ENV PYTHON_PATH=/usr/local/bin/python3.10

# Copy project files with correct ownership
COPY --chown=vscode:vscode ../ /workspace

# Convenience: start in /workspace on shell open
RUN echo 'cd /workspace' >> /home/vscode/.bashrc

# Install only build tools globally; project deps go into tox env
RUN $PYTHON_PATH -m pip install --no-cache-dir --upgrade pip wheel virtualenv tox \
  && $PYTHON_PATH -m pip install "virtualenv==20.35.3"

# Set tox work directory in user's home to avoid volume mount conflicts
ENV TOX_WORK_DIR=/home/vscode/.tox-cache

# Create tox cache directory with proper ownership
RUN mkdir -p $TOX_WORK_DIR && chown -R vscode:vscode $TOX_WORK_DIR

# Pre-build the single shared tox env (creates /home/vscode/.tox-cache/python310)
# Assumes your tox.ini defines an env named "python310" that installs all deps.
RUN $PYTHON_PATH -m tox --workdir $TOX_WORK_DIR -e python310 --notest -vv || true

# Auto-activate the shared tox env in interactive shells, and ensure tox won't recreate it out from under you
RUN echo '[ -f /home/vscode/.tox-cache/python310/bin/activate ] && source /home/vscode/.tox-cache/python310/bin/activate' \
      >> /home/vscode/.bashrc && \
    echo 'alias tox="$PYTHON_PATH -m tox --workdir /home/vscode/.tox-cache"' >> ~/.bashrc

# Default shell
CMD ["bash", "-c", "cd /workspace && exec bash"]

# Recreate tox inside image
# $PYTHON_PATH -m tox --workdir /home/vscode/.tox-cache -e python310 --recreate -vv
# sudo chown -R vscode:vscode /workspace
