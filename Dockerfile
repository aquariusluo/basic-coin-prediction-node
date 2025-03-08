FROM python:3.11-slim AS project_env

# Install curl and clean up
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

FROM project_env

# Copy application code
COPY . /app/

# Run the app directly with Python
CMD ["python", "-u", "/app/app.py"]
