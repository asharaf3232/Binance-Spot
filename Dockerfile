# Use the compatible Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- THIS IS THE MISSING LINE ---
# Copy the rest of the application code into the container
COPY . .

# The command to run the bot
CMD ["python3", "Binance.py"]

