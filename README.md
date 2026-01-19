# M2 Text Classifier API
A production-ready text classification API built with FastAPI, featuring both classical (TF-IDF + Logistic Regression) and modern (DistilBERT) approaches. Classifies news articles into 4 categories: World, Sports, Business, Sci/Tech.

## 🚀 Features

- **Dual Models**: Baseline (TF-IDF + LogReg) and Transformer (DistilBERT) implementations
- **FastAPI Backend**: Async, auto-documented API with Pydantic validation
- **Docker Ready**: Containerized for easy deployment
- **MLOps Ready**: Linting (Ruff), testing (Pytest), CI/CD (GitHub Actions)
- **High Performance**: ~91% accuracy on AG News test set

## 📋 Requirements

- Python 3.11+
- Docker (for containerized deployment)

## 🏃 Quick Start

### Local Development

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/tracy-ml/m2-text-classifier-api.git
   cd m2-text-classifier-api
   conda env create -f environment.yml
   conda activate m2
   ```

2. **Run API**:
   ```bash
   PYTHONPATH=. uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Apple announces new iPhone"}'
   ```

### Docker Deployment

```bash
docker build -t m2-text-classifier-api .
docker run -p 8000:8000 m2-text-classifier-api
```

## 📖 API Documentation

### Endpoints

- `GET /health` - Health check
- `POST /predict` - Classify text

### Predict Request

```json
{
  "text": "Your news article text here"
}
```

### Predict Response

```json
{
  "label": "Sci/Tech",
  "probabilities": {
    "World": 0.02,
    "Sports": 0.01,
    "Business": 0.15,
    "Sci/Tech": 0.82
  },
  "text": "Your news article text here"
}
```

### Interactive Docs

Visit `http://localhost:8000/docs` for Swagger UI with live testing.

## 🛠 Development

### Training Models

```bash
# Baseline model
python src/ml/train_baseline.py

# Transformer model
python src/ml/train_transformer.py
```

### Testing

```bash
# Linting
ruff check src tests

# Unit tests
PYTHONPATH=. pytest tests/
```

### Project Structure

```
m2-text-classifier-api/
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI app
│   └── ml/
│       ├── train_baseline.py    # TF-IDF + LogReg
│       └── train_transformer.py # DistilBERT
├── models/                  # Saved models
├── tests/                   # Pytest tests
├── .github/workflows/       # CI/CD
├── Dockerfile               # Container config
└── requirements.txt         # Dependencies
```

## 🚢 Deployment

### Docker

```bash
docker build -t m2-text-classifier-api .
docker run -p 8000:8000 m2-text-classifier-api
```

### Cloud Platforms

- **Heroku**: `git push heroku main`
- **AWS/GCP**: Use ECS/Cloud Run with the Docker image
- **Railway/Vercel**: Connect GitHub repo for auto-deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `ruff check` and `pytest`
5. Submit a PR

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- AG News dataset from Hugging Face
- DistilBERT model from Hugging Face Transformers
- FastAPI framework