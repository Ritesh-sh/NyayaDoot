# Environment Setup for Legal AI Service

## Setting up Environment Variables

### 1. Create a `.env` file

In the `legal-ai-service` folder, create a file named `.env` (not `.env.txt` or any other extension).

### 2. Add your API key

Open the `.env` file and add your Together AI API key:

```
TOGETHER_API_KEY=your_actual_api_key_here
```

Replace `your_actual_api_key_here` with your real Together AI API key.

### 3. Get your Together AI API key

1. Go to [Together AI](https://www.together.ai/)
2. Sign up or log in to your account
3. Navigate to your API keys section
4. Copy your API key
5. Paste it in the `.env` file

### 4. Example `.env` file

```
# Together AI API Configuration
TOGETHER_API_KEY=tk_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# Other environment variables can be added here
# DATABASE_URL=your_database_url
# DEBUG=True
```

### 5. Security Notes

- **Never commit your `.env` file to version control**
- The `.gitignore` file is already configured to exclude `.env` files
- Keep your API key secure and don't share it publicly
- If you accidentally expose your API key, regenerate it immediately

### 6. Running the application

After setting up the `.env` file, you can run the application normally:

```bash
python main.py
```

The application will automatically load the API key from the `.env` file.

### 7. Troubleshooting

If you get an error about the API key not being set:

1. Make sure the `.env` file is in the same folder as `main.py`
2. Check that the file is named exactly `.env` (not `.env.txt`)
3. Verify that your API key is correct and active
4. Ensure there are no extra spaces or quotes around your API key

### 8. Alternative: Using environment variables directly

If you prefer to set the environment variable directly in your terminal:

**Windows (PowerShell):**
```powershell
$env:TOGETHER_API_KEY="your_api_key_here"
python main.py
```

**Windows (Command Prompt):**
```cmd
set TOGETHER_API_KEY=your_api_key_here
python main.py
```

**Linux/Mac:**
```bash
export TOGETHER_API_KEY="your_api_key_here"
python main.py
```

However, using the `.env` file is recommended as it's more convenient and secure. 