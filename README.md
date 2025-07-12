# ⚖️ Legal Chatbot (NyayaDoot)

An AI-powered legal assistant for Indian law. Query legal information through an intelligent chatbot interface with advanced semantic search, a modern web UI, and privacy-first design.

---

## 🗂️ Project Structure

```
├── legal-ai-service/       # Python FastAPI backend for AI processing
│   ├── main.py             # Main FastAPI app
│   ├── models/             # Embedding model, FAISS index, and data
│   ├── requirements.txt    # Python dependencies
│   └── requirement1.txt    # (legacy/alt requirements)
│
├── l-frontend/             # React-based frontend (no auth, no DB)
│   ├── public/             # Static files
│   ├── src/                # Source code (components, etc.)
│   └── README.md           # Frontend-specific notes
│
├── .gitignore              # Ignored files and folders
└── README.md               # You're here!
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ritesh-sh/NyayaDoot.git
cd NyayaDoot
```

---

### 2. Download the Model Files

The AI model files are too large for Git. Please download the `models/` folder from the following link:

📦 **[Download models from Google Drive](https://drive.google.com/drive/folders/1N8g-YxJkMSTilm0OZvRzlzwQTYpWW4OH?usp=sharing)**

Place the extracted folder into:

```
legal-ai-service/models/
```

You should now have:

```
legal-ai-service/models/
├── legal_embedding_model/
├── legal_index.faiss
└── legal_sections.pkl
```

---

### 3. Set Up the Python Backend (`legal-ai-service`)

```bash
cd legal-ai-service
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirement1.txt
python main.py             # Runs FastAPI backend (default: http://localhost:8000)
```

---

### 4. Set Up the Frontend (`l-frontend`)

```bash
cd l-frontend
npm install
```

#### Configure FastAPI Backend URL (Optional)
- Create a `.env` file in `l-frontend`:
  ```
  VITE_API_URL=http://localhost:8000
  ```
- The frontend will use this URL for all chat requests.

```bash
npm run dev
```
Your frontend will be live at: `http://localhost:5173`

---

## 💡 Features

* 🧠 **Semantic Legal Search** — Find relevant Indian legal information instantly using advanced AI-powered semantic search.
* 💬 **Intelligent Legal Chatbot** — Ask legal questions and get instant, AI-powered answers tailored to Indian law.
* 🔐 **Simple Captcha Verification** — No login or registration required. Just solve a simple captcha to start chatting.
* 🕑 **Session-Based Chat History** — Your chat history is stored only in your browser for privacy and is cleared when you close the tab.
* 🛡️ **Privacy-Focused** — No user data or chat history is stored on any server. Everything stays on your device.
* ⚡ **Modern, Accessible UI** — Beautiful, responsive, and accessible interface for all users.

---

## 🔒 Environment Variables

- **Backend:**
  - Set `TOGETHER_API_KEY` in your environment for the FastAPI backend (see `main.py`).
- **Frontend:**
  - Set `VITE_API_URL` in `l-frontend/.env` if your backend is not on `localhost:8000`.

---

## 📦 Dependencies

* Python: `FastAPI`, `SentenceTransformers`, `FAISS`, `Pickle`, `Together`
* React: `React Router`, `MUI`

---

## 🙋‍♀️ Questions or Contributions?

Feel free to open issues or submit pull requests. Contributions are welcome!
