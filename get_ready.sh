 #/bin/bash
sudo docker run -d --name model-api -p8080:5000 modenaf360/sentiment-detector

sudo docker logs -f model-api
