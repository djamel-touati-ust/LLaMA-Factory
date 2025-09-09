FROM python:3.10-slim

# Default export paths for configs and datasets
ENV LLAMAFAC_CONFIG_DIR=/work/configs \
    LLAMAFAC_DATA_DIR=/work/datasets \
    GRADIO_SERVER_PORT=7860

RUN mkdir -p "$LLAMAFAC_CONFIG_DIR" "$LLAMAFAC_DATA_DIR" && \
    apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight requirements
COPY requirements-export.txt /app
RUN pip install --no-cache-dir -r requirements-export.txt

# Copy source and install LLaMA Factory without extra deps
COPY . /app
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 7860
CMD ["bash", "-lc", "llamafactory-cli webui --host 0.0.0.0 --port 7860"]
