FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY launcher/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY launcher/app /app/app
COPY scripts/retire_workspace_exercises.py /app/migrations/retire_workspace_exercises.py
COPY participant/workspace/start_here.ipynb /app/templates/start_here.ipynb
COPY participant/workspace/README.md /app/templates/README.md
COPY participant/workspace/xray_blue/blue_starter.ipynb /app/templates/xray_blue/blue_starter.ipynb
COPY participant/workspace/xray_blue/blue_team.ipynb /app/templates/xray_blue/blue_team.ipynb
EXPOSE 7000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
