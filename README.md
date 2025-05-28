
# 🤖 Response Inquiry

A smart AI-powered assistant that classifies customer inquiries into:

- 🛠 Technical Support
- 💡 Feature Requests
- 💼 Sales Leads

…and responds intelligently using a trained machine learning model.

---

## 📁 Project Structure

```
nullaxis1/
├── client/                     # React frontend
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── InquiryForm.jsx
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   └── README.md
│
├── server/                     # FastAPI backend
│   ├── main.py                 # FastAPI app with classification logic
│   ├── train_model.py          # Script to train the ML model
│   ├── model.joblib            # Trained classifier
│   ├── bert_model.joblib       # (optional) BERT embeddings
│   ├── label_encoder.joblib    # Encodes target labels
│   ├── requirements.txt        # Python dependencies
│   └── venv/                   # Virtual environment (not pushed to GitHub)
└── README.md                   # This file
```

---

## 🚀 Backend Setup (FastAPI + ML Model)

### 1. Navigate to the backend folder

```bash
cd server
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

- Server will run at: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

---

## 🌐 Frontend Setup (React)

### 5. Navigate to the frontend folder

```bash
cd ../client
```

### 6. Install frontend dependencies

```bash
npm install
```

### 7. Start the React app

```bash
npm start
```

- App runs at: http://localhost:3000

> Ensure the backend is running at port 8000 before using the app.

---

## 🧠 Train the Model (Optional)

If you'd like to retrain the ML model from scratch:

```bash
cd server
python train_model.py
```

- This will regenerate:  
  `model.joblib`, `bert_model.joblib`, and `label_encoder.joblib`.

---



