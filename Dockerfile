FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install Python 3.9 and necessary packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-venv && \
    apt-get install -y python3-pip python3-dev && \
    apt-get clean

# Set python3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install dependencies
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchdiffeq wandb numpy pandas matplotlib

# Set the working directory
WORKDIR /experiment

# Copy the local directory contents to the container
COPY . /experiment