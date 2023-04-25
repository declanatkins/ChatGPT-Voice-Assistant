FROM tverous/pytorch-notebook:latest


ARG DEBIAN_FRONTEND=noninteractive


# Install dependencies
RUN apt-get update && apt-get install -qq -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    portaudio19-dev


# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt