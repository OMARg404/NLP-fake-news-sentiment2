from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =======================
# 🟢 تحديد المسارات الصحيحة
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)  # المجلد الأب (يحتوي على مجلد models)
MODELS_DIR = os.path.join(PARENT_DIR, "models")  # مسار مجلد models

# =======================
# 🟢 تحميل المودلز من مجلد models
# =======================
models_loaded = {}

try:
    # تحميل موديلات ML التقليدية من مجلد models
    naive_bayes_path = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
    if os.path.exists(naive_bayes_path):
        models_loaded['naive_bayes'] = joblib.load(naive_bayes_path)
        models_loaded['bow_vectorizer'] = joblib.load(os.path.join(MODELS_DIR, "bow_vectorizer.pkl"))
        print("✅ تم تحميل Naive Bayes")
    
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_best_model.pkl")
    if os.path.exists(tfidf_path):
        models_loaded['tfidf_model'] = joblib.load(tfidf_path)
        models_loaded['tfidf_vectorizer'] = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        print("✅ تم تحميل TF-IDF Model")
    
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    if os.path.exists(lstm_path):
        models_loaded['lstm_model'] = load_model(lstm_path)
        models_loaded['tokenizer'] = pickle.load(open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb"))
        print("✅ تم تحميل LSTM Model")
    
except Exception as e:
    print(f"❌ خطأ في تحميل بعض الموديلات: {e}")

# =======================
# 🟢 تحميل موديل Transformers من المجلد الحالي (backend)
# =======================
try:
    # الملفات الموجودة في مجلد backend (الحالي)
    transformer_tokenizer = AutoTokenizer.from_pretrained(
        BASE_DIR,  # المسار الحالي حيث توجد ملفات tokenizer
        local_files_only=True
    )
    
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_DIR,  # المسار الحالي حيث توجد ملفات النموذج
        local_files_only=True
    )
    transformer_model.eval()
    
    print("✅ تم تحميل نموذج Transformers بنجاح!")
    
except Exception as e:
    print(f"❌ خطأ في تحميل نموذج Transformers: {e}")
    transformer_tokenizer = None
    transformer_model = None

# =======================
# 🔹 تعريف FastAPI
# =======================
app = FastAPI(title="Fake News Detection API")

# 🔹 CORS Middleware للسماح للـReact frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "*"  # للتطوير فقط
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# 📝 شكل الداتا اللي هيجيلنا من الـ frontend
# =======================
class InputText(BaseModel):
    text: str

# =======================
# 🔹 دوال التنبؤ لكل موديل
# =======================
def predict_naive_bayes(text: str):
    try:
        X = models_loaded['bow_vectorizer'].transform([text])
        pred = models_loaded['naive_bayes'].predict(X)[0]
        return "Fake" if pred == 1 else "Real"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_tfidf(text: str):
    try:
        X = models_loaded['tfidf_vectorizer'].transform([text])
        pred = models_loaded['tfidf_model'].predict(X)[0]
        return "Fake" if pred == 1 else "Real"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_lstm(text: str):
    try:
        seq = models_loaded['tokenizer'].texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        pred = models_loaded['lstm_model'].predict(padded)
        return "Fake" if pred[0][0] > 0.5 else "Real"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_transformer(text: str):
    try:
        inputs = transformer_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        return "Fake" if pred == 1 else "Real"
    except Exception as e:
        return f"Error: {str(e)}"

# =======================
# 🔹 API Endpoint للتنبؤ
# =======================
@app.post("/predict")
def predict_endpoint(input: InputText):
    text = input.text.strip()
    
    if not text:
        return {"error": "النص فارغ"}
    
    results = []
    
    # 🔹 إضافة نتائج جميع الموديلات المتاحة
    if 'naive_bayes' in models_loaded:
        results.append({"model": "Naive Bayes (BoW)", "prediction": predict_naive_bayes(text)})
    
    if 'tfidf_model' in models_loaded:
        results.append({"model": "TF-IDF Model", "prediction": predict_tfidf(text)})
    
    if 'lstm_model' in models_loaded:
        results.append({"model": "LSTM", "prediction": predict_lstm(text)})
    
    if transformer_model is not None:
        results.append({"model": "Transformers (DistilBERT)", "prediction": predict_transformer(text)})
    
    # 🔹 حساب التصويت النهائي (average vote)
    # تحويل "Fake" → -1 و "Real" → +1
    scores = []
    for r in results:
        if r['prediction'] == "Fake":
            scores.append(-1)
        elif r['prediction'] == "Real":
            scores.append(1)
    
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.1:  # الأغلبية إيجابية
            final_sentiment = "Positive"
        elif avg_score < -0.1:  # الأغلبية سلبية
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"
    else:
        final_sentiment = "Neutral"

    return {
        "input_text": text,
        "results": results,
        "final_sentiment": final_sentiment
    }


# =======================
# 🔹 endpoints إضافية
# =======================
@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running!"}

@app.get("/health")
def health_check():
    models_status = {
        "naive_bayes": "loaded" if 'naive_bayes' in models_loaded else "not available",
        "tfidf_model": "loaded" if 'tfidf_model' in models_loaded else "not available", 
        "lstm_model": "loaded" if 'lstm_model' in models_loaded else "not available",
        "transformer_model": "loaded" if transformer_model is not None else "not available"
    }
    return {
        "status": "ok",
        "models_loaded": models_status
    }

@app.get("/models")
def list_models():
    available_models = []
    if 'naive_bayes' in models_loaded:
        available_models.append("Naive Bayes")
    if 'tfidf_model' in models_loaded:
        available_models.append("TF-IDF")
    if 'lstm_model' in models_loaded:
        available_models.append("LSTM")
    if transformer_model is not None:
        available_models.append("Transformers")
    
    return {"available_models": available_models}

# =======================
# 🔹 تشغيل السيرفر
# =======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)