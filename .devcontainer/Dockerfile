FROM nvidia/cuda:13.0.2-base-ubuntu24.04

# --- System dependencies ---
RUN apt-get update && apt-get install -y\
    git curl wget python3 python3-venv python3-dev build-essential\
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- Setup project ---
WORKDIR /app

# Copy project files
COPY src/inference/resources/ /app/


# --- Install PyTorch matching CUDA version ---
RUN uv venv && \
    uv pip install torch==2.9.1+cu130 torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# Set venv as default Python
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# --- Install project with inference extras ---
RUN uv pip install -e ".[inference]"

# --- Expose port for vLLM server mode ---
EXPOSE 8000

# --- Default command: run vLLM as an OpenAI-compatible server ---
ENTRYPOINT ["vllm", "serve", "google/gemma-3-1b-it", "--dtype", "bfloat16", "--max-lora-rank", "32", "--host", "0.0.0.0", "--logprobs-mode", "processed_logprobs", "--enable-lora", "--lora-modules", "stage-1-classifier=[LoRA-dir]"]
