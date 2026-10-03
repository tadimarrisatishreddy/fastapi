# Spam Detection API

A full-stack spam detection app built with **FastAPI** (Python backend) and a clean HTML/CSS/JS frontend.

## 🔍 How It Works

1. A **Logistic Regression** model is trained on the [SMS Spam Collection dataset](https://archive.uci.edu/dataset/228/sms+spam+collection) using TF-IDF features.
2. The trained model (`spam_model.joblib`) and vectorizer (`vectorizer.joblib`) are saved and loaded by the FastAPI server at startup.
3. The frontend (`frontend/ui/index.html`) sends messages to `/predict` and displays the result.

## 📁 Project Structure

```
fastapi/
├── frontend/
│   ├── main.py            # FastAPI app (API + serves the UI)
│   ├── spamdetection.py   # Model training script
│   ├── Evalautemodel.py   # Model evaluation script
│   └── ui/
│       └── index.html     # Frontend UI
├── spam_model.joblib      # Trained model (binary)
├── vectorizer.joblib      # TF-IDF vectorizer (binary)
├── requirements.txt
├── vercel.json            # Vercel deployment config
└── pyproject.toml
```

## 🚀 Run Locally

```bash
pip install -r requirements.txt
uvicorn frontend.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 🌐 API Endpoints

| Method | Path            | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/`             | Serves the frontend UI             |
| GET    | `/health`       | Health check                       |
| POST   | `/predict`      | Classify a single message          |
| POST   | `/predict-batch`| Classify multiple messages at once |
| GET    | `/docs`         | Interactive Swagger UI             |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a free iPhone. Click here."}'
```

### Example Response

```json
{
  "message": "Congratulations! You won a free iPhone. Click here.",
  "prediction": "SPAM",
  "spam_probability": 0.9871
}
```

## ☁️ Deploy on Vercel

This project is configured for Vercel via `vercel.json`. Just connect the GitHub repo to Vercel and it deploys automatically.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)
