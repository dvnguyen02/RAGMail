# RAGMail: Email Management with RAG

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-development-orange)

# RAGMail

RAGMail is an intelligent email management system that uses Retrieval Augmented Generation (RAG) to help you search, analyze, and interact with your emails using LLM.

## Features

- **Smart Email Search**: Find emails using natural language queries
- **Ask Questions About Your Emails**: Get answers based on your email content
- **Daily Email Summary**: Automatically get a summary of recent emails when you start your computer
- **Interactive Mode**: Converse naturally with your email data

The system can operate in multiple search modes:
- **Semantic Search**: Using vector embeddings for similarity matching
- **Keyword Search**: Traditional keyword-based search
- **Hybrid Search**: Combines both approaches for better results

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/dvnguyen02/RAGMail.git
   cd RAGMail
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```
   
3. Create a `.env` file in the project root with the following configuration:
   ```
   # Email credentials
   GMAIL_USERNAME=your_gmail@gmail.com
   GMAIL_PASSWORD=your_app_password

   # OpenAI API key
   OPENAI_API_KEY=your_openai_api_key
   
   # Optional configuration
   OPENAI_MODEL=your_gpt_model
   EMBEDDING_MODEL=all-MiniLM-L6-v2 (recommended this embedding model, or you could use other private embedding models)
   STORAGE_PATH=ragmail_data
   API_RATE_LIMIT=10 (to control the api uses)

   # App settings
   DEFAULT DAYS BACK = 30
   DEFAULT_TOP_K = 5
   DEFAULT_TEMPERATURE = 0.6
   ```

   **Note**: For Gmail, we advise you to use an App Password instead of your actual password. Click here to find where your app password located [App Password](https://support.google.com/accounts/answer/185833?hl=en) 

## Usage

### Interactive Mode

The easiest way to use RAGMail is through its interactive mode:

```
python ragmail.py interactive
```

This will start an interactive shell where you can:
- Search for emails: `search [query]`
- Ask questions: `ask [question]` or simply type your question
- Sync more emails: `sync [limit]`
- View email summary: `summary [days]` (defaults to 1 day)
- Get help: `help`
- Exit: `exit`

### Command Line Usage

You can also use specific commands directly:

1. Sync recent emails:
   ```
   python ragmail.py sync --limit 20
   ```
   After syncing you could use the services. 
   * Note that if you sync too much it might cause the LLM to hallucinate.

2. Search for emails:
   ```
   python ragmail.py search "meeting with clients"
   ```

3. Ask a question about your emails:
   ```
   python ragmail.py ask "When is my next appointment?"
   ```

4. Get a summary of recent emails:
   ```
   python ragmail.py summary --days 1
   ```

### Automatic Email Summary on Boot [NEW UPDATE]

For a daily briefing of your emails when you start your computer:

#### Windows

1. Create a batch file (e.g., `RunRAGMail.bat`) with the following content:
   ```batch
   @echo off
   cd C:\path\to\your\ragmail\directory
   python LaunchRagMail.py
   pause
   ```

2. Add to startup:
   - Press `Win + R` and type `shell:startup`
   - Create a shortcut to your batch file in this folder

## Your Email Data

- In order to see what emails has been synced to your computer, after you run the sync feature, there will be a ragmail_data directory which comprises two sub dir: 
- email_data (json format) and vector_data (vector)

## Development

## Future Enhancements

- Support for Outlook
- Agentic Features ( automatically reply to an email )

## License

This project is licensed under the MIT License - see the LICENSE file for details.
