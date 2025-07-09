# Legal Chatbot Frontend (l-frontend)

## Overview
This is the React frontend for the Legal Chatbot project. It now works without any Node.js backend or database. All chat processing is handled by the FastAPI backend (`legal-ai-service`).

## Features
- **Landing Page**: Entry point for users.
- **Captcha Page**: Simple math captcha to verify the user (no login or registration).
- **Chat Page**: Users can chat with the legal AI. Chat history is stored in the browser's sessionStorage and is cleared when the tab or browser is closed.
- **No authentication or user accounts**: All previous login/register/database logic has been removed.

## Architecture & Flow
1. **Landing Page** (`/`): User starts here and clicks to proceed.
2. **Captcha Page** (`/captcha`): User must solve a simple math captcha to continue.
3. **Chat Page** (`/chat`): User can chat with the AI. Each message is sent to the FastAPI backend for processing. Chat history is kept only in the browser (sessionStorage).

### How the FastAPI Backend is Called
- The frontend sends chat queries to the FastAPI backend at:
  ```
  POST http://localhost:8000/process-query
  ```
- The backend processes the query and returns a response, which is shown in the chat.
- The backend is **not** used for authentication, user management, or chat history.

## Configuring the FastAPI Backend URL
By default, the frontend sends requests to `http://localhost:8000/process-query`.

To change the backend URL (for deployment or different environments):
1. Create a `.env` file in the root of `l-frontend`.
2. Add:
   ```
   VITE_API_URL=http://your-fastapi-backend:8000
   ```
3. In your code, use `import.meta.env.VITE_API_URL` to reference the backend URL (see below for code update).

## Updating the Code to Use the Environment Variable
In your API calls (e.g., in `Chat.jsx`), replace:
```js
'http://localhost:8000/process-query'
```
with
```js
`${import.meta.env.VITE_API_URL}/process-query`
```

## Running the Frontend
1. Install dependencies:
   ```
   npm install
   ```
2. Start the dev server:
   ```
   npm run dev
   ```

## Running the Backend
See the `legal-ai-service/README.md` for backend instructions.

## No Node Backend Required
You can delete the `legal-backend` folder and all related files. The project is now fully decoupled from Node.js and MySQL.
