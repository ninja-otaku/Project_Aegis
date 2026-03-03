FROM python:3.11-slim

# OpenCV requires these two system libraries even in headless mode.
# libgl1-mesa-glx provides libGL.so.1; libglib2.0-0 provides libgobject.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
# independently of source-code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source.
COPY . .

EXPOSE 8765

CMD ["python", "main.py"]
