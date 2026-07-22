# Base image with Python 3.10
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside container
WORKDIR /app

# Install dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run uvicorn server binding to 0.0.0.0 for external container access
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
