# Use a newer, compatible Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install a comprehensive set of build tools for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    liblapack-dev \
    libblas-dev \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# The command to run the bot (Change Binance.py if the name is different)
CMD ["python3", "Binance.py"]
