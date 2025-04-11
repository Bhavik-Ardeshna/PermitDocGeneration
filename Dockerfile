# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Install dependencies in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*  
# Removes package lists to reduce docker image size

WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Now copy the rest of the app's source code
COPY . /app

# Set the NLTK data path to this directory
ENV NLTK_DATA /app/nltk_data

# Download transformer models to perform vector tasks
RUN python /app/download_model_docker.py

# Set the port the app runs on
ENV PORT 8080
ENV NUM_WORKERS 2
ENV TIMEOUT 3600
# Run the application.
EXPOSE $PORT

# Command to run the application
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT" --workers $NUM_WORKERS]
 CMD ["sh", "-c", "uvicorn main:app --host=0.0.0.0 --port=$PORT"]
#CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w $NUM_WORKERS -b 0.0.0.0:$PORT --timeout $TIMEOUT main:app"]
