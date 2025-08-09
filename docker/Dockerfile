# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 wget git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ../requirements.txt .
COPY ../src ./src
COPY ../configs ./configs

# Install Python packages
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 && \
    pip install -r requirements.txt


