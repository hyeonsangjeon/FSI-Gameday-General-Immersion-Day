FROM tensorflow/tensorflow:2.3.0-gpu
MAINTAINER your_name "wingnut0310@gmail.com"

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN rm -rf /var/lib/apt/lists/*


COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip


COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8080

#ENTRYPOINT ["python"]
CMD ["python", "/app/happy_emotional_flask.py"]

