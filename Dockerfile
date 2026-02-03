# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages like numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install CPU-only PyTorch (Significantly reduces image size)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install any needed packages specified in requirements.txt
# Install any needed packages specified in requirements.txt
# Split large install into chunks to save temp space
RUN pip install --no-cache-dir fastapi uvicorn motor python-multipart && \
    pip install --no-cache-dir python-jose[cryptography] passlib[bcrypt] && \
    pip install --no-cache-dir sentence-transformers && \
    pip install --no-cache-dir langchain langchain-community && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the current directory contents into the container at /app
COPY backend ./backend
COPY .env .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV MODULE_NAME="backend.main"
ENV VARIABLE_NAME="app"
ENV PORT=8000

# Run uvicorn when the container launches
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
