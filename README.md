# KoBERT Sentiment Analysis

한국어 문장 감정 분석 (긍정/부정) 서비스. KoBERT 기반 이진 분류기를 FastAPI로 서빙합니다.

![Architecture](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/simple_architecture.png?raw=true)

## Quick Start

### Docker (권장)

```bash
docker build -t kobert-sentiment .
docker run -p 5000:5000 kobert-sentiment
```

→ http://localhost:5000

GPU 사용:
```bash
docker run -p 5000:5000 --gpus all kobert-sentiment
```

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

→ http://localhost:5000

## API

### POST /predict

한국어 문장의 감정을 분석합니다.

**Request** (form data):
```bash
curl -X POST http://localhost:5000/predict -d "input_data=오늘 기분이 좋다"
```

**Response**:
```json
{
  "text": "오늘 기분이 좋다",
  "sentiment": "긍정",
  "confidence": 92.34,
  "raw_score": 0.9234
}
```

### GET /docs

Swagger UI (FastAPI 자동 생성)

### GET /health

서버 상태 확인:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Python 예제

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    data={"input_data": "오늘 정말 행복한 하루였어요"},
)
print(response.json())
```

### JavaScript 예제

```javascript
const fd = new FormData();
fd.append("input_data", "오늘 정말 행복한 하루였어요");

const res = await fetch("http://localhost:5000/predict", {
  method: "POST",
  body: fd,
});
console.log(await res.json());
```

## Architecture

- **Framework**: FastAPI + Uvicorn
- **Model**: KoBERT fine-tuned for binary sentiment classification
- **Model Source**: [HuggingFace Hub](https://huggingface.co/HyeonSang/kobert-sentiment) (`HyeonSang/kobert-sentiment`)
- **Tokenizer**: SentencePiece (KoBertTokenizer)
- **Deep Learning**: TensorFlow 2.x
- **GPU/CPU**: Docker 환경에서 자동 GPU 감지, 로컬에서는 기본 전략 사용

```
Client → FastAPI (app.py)
              │
              ▼
        SentimentPredictor (model.py)
              │
         ┌────┴────┐
         ▼         ▼
  KoBertTokenizer  TFBertModel + Dense
  (tokenizer.py)   (BertConfig → load_weights)
```

## Project Structure

```
├── app.py                  # FastAPI 진입점
├── model.py                # SentimentPredictor 클래스 (모델 로딩 + 추론)
├── tokenizer.py            # KoBertTokenizer 클래스
├── static/
│   └── index.html          # Tailwind CSS 기반 웹 UI
├── Dockerfile              # uvicorn CMD
├── requirements.txt        # Python 패키지 (8개)
├── .gitignore
├── README.md
├── CLAUDE.md
└── LICENSE                 # MIT
```

## 권장 사양

- **CPU**: 4 vCPU 이상
- **메모리**: 16GB RAM
- **스토리지**: 50GB (Docker 이미지 포함)
- **GPU** (선택): NVIDIA GPU + CUDA

## 문제 해결

### 메모리 부족
Docker에 할당된 메모리를 16GB 이상으로 증가시키세요.

### 모델 로딩 실패
인터넷 연결을 확인하세요. 첫 실행 시 HuggingFace Hub에서 모델을 다운로드합니다 (~370MB).

### 포트 충돌
```bash
lsof -i :5000
docker run -p 8080:5000 kobert-sentiment  # 다른 포트 사용
```

## License

[MIT](LICENSE)

## 참고 자료

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/)
