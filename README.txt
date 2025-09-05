# Fake News Detection Project

This repository contains the backend and frontend for a Fake News Detection system using multiple machine learning and transformer-based models.

⚠️ **Important:** Models are not included in this repo.
You need to download the models separately and place them in the correct paths:

* Traditional ML models (Naive Bayes, TF-IDF, LSTM) → `backend/models/` folder
* Transformers model files → `backend/` folder (same folder as `main.py`)

[Google Drive - All Models](https://drive.google.com/file/d/1wcAJ8sDf36KbvGuZx5jwjEaPi6lJXnC/view?usp=sharing)

---

## ⚙️ Backend Setup

Navigate to the backend folder:

```bash
cd backend
```

Install Python dependencies:

```bash
pip install fastapi uvicorn joblib pickle-mixin tensorflow transformers torch pydantic
```

Run the backend server:

```bash
python main.py
```

The backend will start at `http://localhost:8000` with the following endpoints:

* `POST /predict` → Predict fake/real news using all models
* `GET /` → Root endpoint to check API is running
* `GET /health` → Check loaded models status
* `GET /models` → List available models

> The backend automatically loads all models from their paths. Ensure the paths are correct, otherwise some models may fail to load.

---

## 🖥️ Frontend Setup

The frontend is built with React.

Navigate to the frontend folder (nested):

```bash
cd frontend
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` (or port 3000). It communicates with the backend to send text for prediction and display results from all models.

---

## 📂 Other Important Files

* `notebooks/` → Jupyter notebooks for experimentation
* `data/` → Dataset files
* `FakeNews_Sentiment_Report.docx` → Project report
* `.gitignore` → Git ignore configuration
