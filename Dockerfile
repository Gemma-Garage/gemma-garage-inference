FROM python:3.13-slim

#huggingface token
ARG HF_TOKEN
#Gemini API key
ARG GEMINI_KEY

# Set default environment variables for GCS buckets
ENV NEW_DATA_BUCKET="gs://llm-garage-datasets"
ENV NEW_MODEL_OUTPUT_BUCKET="gs://llm-garage-models/gemma-peft-vertex-output"
ENV NEW_STAGING_BUCKET="gs://llm-garage-vertex-staging"

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN huggingface-cli login --token ${HF_TOKEN}  


# Copy the rest of the application
COPY . .

# Expose only the FastAPI port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
