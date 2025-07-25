FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Install build-essential for C compiler and git
RUN apt-get update && \
    apt-get install -y build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN huggingface-cli login --token ${HF_TOKEN}

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
