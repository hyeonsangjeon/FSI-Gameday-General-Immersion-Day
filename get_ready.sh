 #/bin/bash
 docker run -d --name model-api -p8088:5000 modenaf360/sentiment-detector

 docker logs -f model-api
