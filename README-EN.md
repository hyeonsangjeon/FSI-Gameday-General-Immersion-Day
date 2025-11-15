# Korean Sentiment Analysis API

A TensorFlow BERT-based Korean sentiment analysis REST API service. Built with Flask and deployed as a Docker container.

## Features

- **Korean Sentiment Analysis**: Binary classification (positive/negative) using KoBERT model
- **REST API**: Swagger UI powered by Flask-RESTX
- **Flexible Model Loading**: Load models from HuggingFace Hub or local files
- **Automatic GPU/CPU Selection**: Automatically selects optimal device based on environment
- **Docker Deployment**: Containerized deployment for consistent execution environment

![Architecture](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/simple_architecture.png?raw=true)

## Tech Stack

### Core Framework
- **TensorFlow 2.20.0**: Deep learning framework
- **Transformers 4.57.1**: HuggingFace transformers library
- **Flask 2.2.5**: Web framework
- **Flask-RESTX**: REST API and Swagger documentation

### Model
- **Base Model**: KoBERT (monologg/kobert) - Korean BERT pretrained model
- **Task**: Binary Sentiment Classification (positive/negative)
- **Sequence Length**: 32 tokens
- **Tokenizer**: SentencePiece-based KoBertTokenizer

## Quick Start

### Prerequisites

- Docker installed
- (Optional) NVIDIA GPU with CUDA support (for GPU acceleration)
- Minimum 16GB RAM recommended

### Run with Docker

#### 1. Clone Repository
```bash
git clone https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day.git
cd FSI-Gameday-General-Immersion-Day
```

#### 2. Build Docker Image
```bash
docker build -t sentiment-api .
```

#### 3. Run Container

**Load model from HuggingFace Hub (default)**:
```bash
docker run --name model-api -p 8080:5000 sentiment-api
```

**Use local model file**:
```bash
docker run --name model-api -p 8080:5000 \
  -v $(pwd)/data:/app/data \
  -e USE_LOCAL_MODEL=true \
  sentiment-api
```

**Use GPU**:
```bash
docker run --name model-api -p 8080:5000 --gpus all sentiment-api
```

#### 4. Access API

Open Swagger UI in browser:
```
http://localhost:8080/
```

## API Usage

### Swagger UI

Access `http://localhost:8080/` in your browser to explore the interactive API documentation.

### cURL Example

```bash
# Analyze positive sentence
curl -X POST "http://localhost:8080/Korean%20sentiment%20analysis" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "input_data=오늘 정말 행복한 하루였어요"

# Response example
"긍정적인 문장입니다. 긍정 확률 : [ 93.22% ]"
# Translation: "It's a positive sentence. Positive probability: [ 93.22% ]"
```

### Python Example

```python
import requests

url = "http://localhost:8080/Korean%20sentiment%20analysis"
data = {"input_data": "오늘 정말 행복한 하루였어요"}

response = requests.post(url, data=data)
print(response.json())
```

### JavaScript Example

```javascript
const url = 'http://localhost:8080/Korean%20sentiment%20analysis';
const data = new URLSearchParams({ input_data: '오늘 정말 행복한 하루였어요' });

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: data
})
  .then(response => response.json())
  .then(result => console.log(result));
```

## Model Loading Options

### HuggingFace Hub (Default)

Running without environment variables automatically downloads the model from HuggingFace Hub.

- **Repository**: `HyeonSang/kobert-sentiment`
- **Model File**: `tf_model.h5`
- **Advantages**: No need to manage model files separately, automatic download

```bash
docker run --name model-api -p 8080:5000 sentiment-api
```

### Local File

To use a locally stored model file, set the environment variable.

- **Path**: `./data/fsi_comment_sentiment_model.h5`
- **Advantages**: Works offline, can use custom models

```bash
docker run --name model-api -p 8080:5000 \
  -v $(pwd)/data:/app/data \
  -e USE_LOCAL_MODEL=true \
  sentiment-api
```

## Local Development

### Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python happy_emotional_flask.py
```

### Key Files

- `happy_emotional_flask.py`: Flask REST API server
- `hsjeon_datascience_nlp.py`: NLP utilities and KoBertTokenizer
- `requirements.txt`: Python package dependencies
- `Dockerfile`: Docker image definition
- `get_ready.sh`: Deployment script (EC2)
- `install_back.sh`: Docker installation script (Amazon Linux 2)

## Docker Management

### View Container Logs
```bash
docker logs -f model-api
```

### Stop Container
```bash
docker stop model-api
```

### Remove Container
```bash
docker rm model-api
```

### Remove Image
```bash
docker rmi sentiment-api
```

## Performance Optimization

### GPU Acceleration

If you have an NVIDIA GPU, it will be automatically detected and used:

```bash
docker run --name model-api -p 8080:5000 --gpus all sentiment-api
```

The application automatically switches to GPU mode when `/app` directory exists inside Docker.

### Recommended Specifications

- **CPU**: Minimum 4 vCPU (t2.xlarge or higher)
- **Memory**: 16GB RAM
- **Storage**: 50GB (Docker image size: ~7GB)
- **GPU** (optional): NVIDIA GPU with CUDA support

## Error Handling

The application provides the following error handling:

- **404 Not Found**: Invalid endpoint access
- **400 Bad Request**: Invalid request parameters
- **500 Internal Server Error**: Server internal error (full traceback returned)

All errors are logged to terminal and `./log_data.log` file.

## EC2 Deployment

For deployment on Amazon Linux 2:

### 1. Install Docker
```bash
git clone https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day.git
cd FSI-Gameday-General-Immersion-Day
chmod u+x ./*.sh
./install_back.sh
# Reconnect terminal session required
```

### 2. Run API
```bash
./get_ready.sh
```

### 3. Security Group Settings
- Add TCP port 8080 to inbound rules
- Source: Your IP or required IP range

### 4. Access API
```
http://{EC2_PUBLIC_IP}:8080/
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP POST
       │ /Korean sentiment analysis
       ▼
┌─────────────────────────────┐
│   Flask REST API            │
│   (happy_emotional_flask.py)│
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Text Processing           │
│   (KoBertTokenizer)         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   BERT Model                │
│   (TFBertModel + Dense)     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Sentiment Prediction      │
│   (Positive/Negative)       │
└─────────────────────────────┘
```

## License

This project is provided for educational and research purposes.

## Troubleshooting

### Out of Memory Error
- Increase memory allocated to Docker to 16GB or more
- Remove unnecessary containers and images

### Model Loading Failure
- Check internet connection (when using HuggingFace Hub)
- Verify local model file path
- Check Docker volume mount

### Port Conflict
```bash
# Check process using port 8080
lsof -i :8080
# Use different port
docker run --name model-api -p 9090:5000 sentiment-api
```

## Contributing

Issues and PRs are always welcome.

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Flask-RESTX](https://flask-restx.readthedocs.io/)
- [TensorFlow](https://www.tensorflow.org/)

## Workshop Materials

AWS FSI General Immersion Day workshop materials are available at:
- [Workshop Guide](https://catalog.us-east-1.prod.workshops.aws/workshops/f3a3e2bd-e1d5-49de-b8e6-dac361842e76/ko-KR/preparation-guide/20-event-engine)
