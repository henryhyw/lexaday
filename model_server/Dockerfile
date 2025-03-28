FROM python:3.10-slim

# Turn off bytecode and buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# We assume you want HF_HOME set if you mount a volume at /root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]