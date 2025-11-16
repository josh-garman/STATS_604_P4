# ---- Base image ----
# Lean Jupyter stack with conda & a non-root user (NB_UID/NB_GID).
FROM jupyter/minimal-notebook:latest

# Use bash for all RUN steps (lets us use bash-isms safely)
SHELL ["/bin/bash", "-lc"]

# ---- OS packages (need root) ----
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        make \
        curl \
        git \
        unzip \
        tar \
    && rm -rf /var/lib/apt/lists/*

# ---- Back to unprivileged user ----
USER ${NB_UID}

# Speed up Python layer a bit and keep the image smaller
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ---- Python dependencies ----
# Pin versions in production; unpinned here for brevity.
RUN pip install --no-cache-dir numpy pandas matplotlib

# ---- Project files ----
WORKDIR /app
# Copy your repo into the image and ensure the non-root user owns it
COPY --chown=${NB_UID}:${NB_GID} . /app

# If you have shell scripts, make them executable (no-op if none)
RUN if compgen -G "scripts/*.sh" > /dev/null; then chmod +x scripts/*.sh; fi

# ---- Runtime behavior ----
# Start a bash shell when the container runs.
# CMD clears the base image's default ("start-notebook.sh"),
# so we don't accidentally launch Jupyter.
CMD ["/usr/bin/env", "bash", "-l"]


