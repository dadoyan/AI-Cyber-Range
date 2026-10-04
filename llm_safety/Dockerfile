FROM python:3.11-slim
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /app
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY llm_safety/requirements.txt /tmp/llm-safety-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0 --index-url ${PYTORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r /tmp/llm-safety-requirements.txt
COPY llm_safety/ /app/llm_safety/
EXPOSE 8000
CMD ["uvicorn", "llm_safety.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
