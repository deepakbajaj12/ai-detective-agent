FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install minimal Python deps
COPY core-req.txt /app/core-req.txt
RUN pip install --no-cache-dir -r core-req.txt \
    && pip install --no-cache-dir rq redis

# Copy project (in compose we mount volume for live dev)
COPY . /app

EXPOSE 5000

CMD ["python", "-m", "src.api"]
