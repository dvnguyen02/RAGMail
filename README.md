# RAGMail: Email Management with RAG

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-development-orange)

# RAGMail

RAGMail is an intelligent email management system that uses Retrieval Augmented Generation (RAG) to help you search, analyze, and interact with your emails using natural language.

- **Email Synchronization**: Connect to Gmail and automatically sync your recent emails
- **Semantic Search**: Find relevant emails using natural language queries
- **Question Answering**: Ask questions about your emails and get AI-powered answers
- **Interactive Mode**: Use a command-line interface to interactively work with your emails

## Architecture

RAGMail implements a full Retrieval Augmented Generation (RAG) pipeline:

1. **Retrieval**: Emails are indexed with vector embeddings generated using the Sentence Transformers library. When a query is received, the system performs semantic search to retrieve the most relevant emails based on vector similarity.

2. **Augmentation**: The retrieved emails are added to the context when preparing the prompt for the language model. This provides the LLM with relevant information specific to the user's query.

3. **Generation**: OpenAI's models are used to generate natural language responses based on the augmented context, providing answers that are grounded in the actual content of the user's emails.

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
   OPENAI_MODEL=gpt-3.5-turbo
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   STORAGE_PATH=ragmail_data
   API_RATE_LIMIT=10

   # App settings
   DEFAULT DAYS BACK = 30
   DEFAULT_TOP_K = 5
   DEFAULT_TEMPERATURE = 0.6
   ```

   **Note**: For Gmail, We advise you to use an App Password instead of your actual password. Click here to find where your app password located [App Password](https://support.google.com/accounts/answer/185833?hl=en) 

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
- Get help: `help`
- Exit: `exit`

### Command Line Usage

You can also use specific commands directly:

1. Sync recent emails:
   ```
   python ragmail.py sync --limit 20
   ```
   After syncing you could just use services.

2. Search for emails:
   ```
   python ragmail.py search "meeting with clients"
   ```

3. Ask a question about your emails:
   ```
   python ragmail.py ask "When is my next appointment?"
   ```

## Development

## Future Enhancements

- Support for additional email providers - Outlook
- Integrate a GUI for better experience
- Email categorization and tagging
- Scheduled summaries of important emails (like everyday newletter)
- Advanced filtering and organization capabilities 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
