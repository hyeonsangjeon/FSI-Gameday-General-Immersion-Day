FROM --platform=linux/amd64 tensorflow/tensorflow:2.18.0-gpu

LABEL maintainer="wingnut0310@gmail.com"

RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update -y && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python3", "/app/happy_emotional_flask.py"]