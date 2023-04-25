# Dockerfile for the remote server

FROM python:3.8-slim-buster


# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install gunicorn


# Copy the application
COPY source/ /app
WORKDIR /app


# Run the application
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-t", "360", "remote_api:app"]
