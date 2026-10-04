FROM python:3.11-slim
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /workspace

# PyTorch's Qwen gradient path compiles Triton support code at runtime.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY participant/requirements.txt /tmp/participant-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
       --index-url ${PYTORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r /tmp/participant-requirements.txt

COPY participant/workspace/ /workspace/
COPY range_device.py /usr/local/lib/python3.11/site-packages/range_device.py
COPY assets/mnist/processed/ /workspace/data/MNIST/processed/
RUN cp /workspace/red_blue_arena.py /workspace/xray_red/red_blue_arena.py \
    && cp /workspace/red_blue_arena.py /workspace/xray_blue/red_blue_arena.py \
    && cp /workspace/llm_safety_optimizer.py /workspace/llm_safety/llm_safety_optimizer.py

EXPOSE 8888
CMD ["sh", "-c", "jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=\"$JUPYTER_TOKEN\" --ServerApp.root_dir=/workspace"]
