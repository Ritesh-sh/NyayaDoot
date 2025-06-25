# ⚖️ Legal Chatbot

An AI-powered legal assistant that enables users to query Indian legal information through an intelligent chatbot interface. It integrates custom-trained legal embeddings, a semantic search engine (FAISS), and a modern web UI with authentication and chat history.

---

## 🗂️ Project Structure

```
├── legal-ai-service/       # Python backend for AI processing
│   ├── main.py             # Main FastAPI app
│   ├── models/             # Embedding model, FAISS index, and data
│   ├── requirements.txt    # Python dependencies
│   └── requirement1.txt    # (possibly unused/legacy requirements)
│
├── legal-backend/          # Node.js backend (e.g., authentication, chat history)
│   ├── database schema.sql # MySQL schema for database setup
│   ├── index.js            # Main server file
│   └── migrations/         # SQL migration files
│
├── legal-frontend/         # React-based frontend
│   ├── public/             # Static files
│   ├── src/                # Source code (components, contexts)
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

### 2. Set Up the Database (MySQL)

A SQL schema file is provided at `legal-backend/database schema.sql`.

**Instructions:**
1. Open MySQL Workbench (or your preferred MySQL client).
2. Open the file `legal-backend/database schema.sql`.
3. Copy and execute the entire schema in your MySQL Workbench to create the required database and tables before running the backend.

**Schema Content:**
```sql
CREATE DATABASE legal_chatbot;
USE legal_chatbot;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
select * from users;
select * from chats;
CREATE TABLE chats (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Add session_id column to chats table
ALTER TABLE chats ADD COLUMN session_id VARCHAR(255) NOT NULL DEFAULT 'default_session';

-- Create index for faster session lookups
CREATE INDEX idx_chats_session_id ON chats(session_id);

-- Update existing records to have unique session IDs
UPDATE chats SET session_id = CONCAT('session_', id) WHERE session_id = 'default_session';

-- Check if chats table exists
CREATE TABLE IF NOT EXISTS chats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Check if session_id column exists
SELECT COUNT(*) INTO @exists 
FROM information_schema.columns 
WHERE table_schema = DATABASE()
AND table_name = 'chats' 
AND column_name = 'session_id';

-- Add session_id column if it doesn't exist
SET @sql = IF(@exists = 0,
    'ALTER TABLE chats ADD COLUMN session_id VARCHAR(255) NOT NULL DEFAULT "default_session"',
    'SELECT "session_id column already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Create index for faster session lookups if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_chats_session_id ON chats(session_id);

-- Update existing records to have unique session IDs if they have default_session
UPDATE chats SET session_id = CONCAT('session_', id) WHERE session_id = 'default_session';
```

---

### 3. Download the Model Files

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

### 4. Set Up the Python Backend (`legal-ai-service`)

```bash
cd legal-ai-service
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirement1.txt
python main.py             # Runs FastAPI backend
```

---

### 5. Set Up the Node.js Backend (`legal-backend`)

```bash
cd legal-backend
npm install
# Create a .env file and add your variables
node index.js
```

---

### 6. Set Up the Frontend (`legal-frontend`)

```bash
cd legal-frontend
npm install
npm start
```

Your frontend will be live at: `http://localhost:3000`

---

## 💡 Features

* 🧠 Semantic legal search using FAISS and Sentence Transformers
* 💬 Intelligent chatbot for legal queries
* 🧾 Chat history with session tracking
* 🔐 Auth system with React context
* ⚡ FastAPI + Node + React stack

---

## 🔒 Environment Variables

Make sure you set the necessary variables in:

* `legal-backend/.env`

Example for `.env` (Node backend):

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=legal_chatbot
JWT_SECRET=3a7d5f9b2e8c1a6f4d9c0b5e2f8a3d7e1c4f9a6d2b8e3f7a5c9d1e0b4f8
FASTAPI_URL=http://localhost:8000
PORT=3001
```

---

## 📦 Dependencies

* Python: `FastAPI`, `SentenceTransformers`, `FAISS`, `Pickle`
* Node.js: `Express`, `Mongoose`, `dotenv`
* React: `React Router`, `Context API`, `MUI`

---

## 🙋‍♀️ Questions or Contributions?

Feel free to open issues or submit pull requests. Contributions are welcome!
