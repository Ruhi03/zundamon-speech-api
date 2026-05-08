FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git git-lfs ffmpeg libsox-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /api

COPY download_models.sh /usr/local/bin/download_models.sh
RUN chmod +x /usr/local/bin/download_models.sh

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .