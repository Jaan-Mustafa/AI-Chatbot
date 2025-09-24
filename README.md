# DSAGPT - AI-Powered Data Structures & Algorithms Tutor

## 🎯 Project Overview

DSAGPT is an intelligent chatbot designed to help students learn Data Structures and Algorithms (DSA) through interactive conversations. Built with advanced AI technologies, it provides personalized tutoring, step-by-step hints, and comprehensive explanations of DSA concepts.

### Key Features

- **🤖 AI-Powered Tutoring**: Uses GPT-4 models for intelligent responses
- **📚 Knowledge Base**: RAG (Retrieval-Augmented Generation) system with DSA resources
- **💡 Smart Hints**: Provides progressive hints without giving away complete solutions
- **🔍 Intent Analysis**: Understands whether users want explanations or problem-solving hints
- **💬 Interactive Chat**: Real-time conversation with follow-up questions
- **🔐 User Authentication**: Google OAuth integration for secure access
- **📊 Usage Tracking**: Monitors user interactions and prompt limits
- **💾 Persistent Chat**: Saves conversation history in PostgreSQL database

## 🏗️ Architecture

The project uses a sophisticated LangGraph-based workflow that intelligently routes user queries:

```
User Message → Intent Analysis → Route Decision
                                    ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
            Missing Info?        Want Hints?      Want Explanation?
                    ↓                 ↓                 ↓
            Ask for Details    Generate Hints    RAG Knowledge Search
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                    ↓
                              Generate Response
```

### Core Components

- **`app.py`**: Main Chainlit application with chat interface
- **`graph.py`**: LangGraph workflow for intelligent routing
- **`tools.py`**: Specialized tools for hints and RAG
- **`database/`**: PostgreSQL models and utilities
- **`config.py`**: Configuration and model settings
- **`utils.py`**: Helper functions for PDF processing and formatting

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- OpenAI API key
- Google OAuth credentials (for authentication)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Chatbot-AI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Database Configuration
   DATABASE_URL_BASE=postgresql://username:password@localhost:5432/database_name
   
   # Google OAuth Configuration
   OAUTH_GOOGLE_CLIENT_ID=your_google_client_id
   OAUTH_GOOGLE_CLIENT_SECRET=your_google_client_secret
   CHAINLIT_AUTH_SECRET=your_random_secret_key
   
   # Optional: User Management
   USER_ID=admin_user_id
   PASSWORD=admin_password
   SALES_CHANNEL_USER_ID=sales_user_id
   ```

5. **Set up PostgreSQL database**
   
   Create a PostgreSQL database and run the following to initialize tables:
   ```python
   from database.db import engine, Base
   Base.metadata.create_all(bind=engine)
   ```

6. **Add DSA resources**
   
   Place your DSA-related PDF files in the `pdfs/` directory. The system will automatically process them for the knowledge base.

### Running the Application

1. **Start the Chainlit server**
   ```bash
   chainlit run app.py
   ```

2. **Access the application**
   
   Open your browser and navigate to `http://localhost:8000`

3. **Authenticate with Google**
   
   Click the Google login button and authorize the application.

## 🎮 Usage

### Basic Interaction

1. **Ask for explanations**: "What is binary search?"
2. **Request hints**: "Give me hints to solve the two sum problem"
3. **Specify programming language**: "How to implement quicksort in Python?"

### Advanced Features

- **Follow-up questions**: Click on suggested follow-up questions for deeper learning
- **Chat history**: Previous conversations are automatically saved and can be resumed
- **Progressive hints**: Get step-by-step guidance without complete solutions

### User Types

- **Free Users**: Limited prompts per session
- **Premium Users**: Extended prompt limits
- **Test Users**: Higher limits for testing purposes

## 🔧 Configuration

### Model Settings (`config.py`)

```python
INTENT_MODEL = "gpt-4o"          # For intent analysis
RAG_MODEL = "gpt-4o"             # For main responses
FOLLOWUP_MODEL = "gpt-4o-mini"   # For follow-up questions
```

### Chainlit Configuration (`chainlit.yaml`)

- OAuth settings for Google authentication
- WebSocket configuration for real-time chat
- Theme and UI preferences

## 📁 Project Structure

```
Chatbot-AI/
├── app.py                 # Main Chainlit application
├── graph.py              # LangGraph workflow
├── tools.py              # Specialized tools
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── chainlit.yaml         # Chainlit configuration
├── gunicorn_config.py    # Production server config
├── database/             # Database layer
│   ├── db.py            # Database connection
│   ├── models.py        # SQLAlchemy models
│   └── db_utils.py      # Database utilities
├── evaluation/           # Evaluation tools
├── pdfs/                 # DSA resource files
├── public/               # Static assets
└── script/               # Utility scripts
```

## 🛠️ Development

### Adding New Features

1. **New Tools**: Add functions to `tools.py` and register them in the graph
2. **Database Changes**: Update models in `database/models.py`
3. **UI Changes**: Modify Chainlit components in `app.py`

### Testing

Run the evaluation tools:
```bash
python evaluation/evaluation.py
python evaluation/tool_evaluation.py
```

## 🚀 Deployment

### Production Setup

1. **Use Gunicorn**:
   ```bash
   gunicorn -c gunicorn_config.py app:app
   ```

2. **Environment Variables**: Ensure all production environment variables are set

3. **Database**: Use a production PostgreSQL instance

4. **Static Files**: Serve static files through a web server like Nginx

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is developed for educational purposes. Please check the license file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Error**: Check PostgreSQL is running and credentials are correct
2. **OpenAI API Error**: Verify API key is valid and has sufficient credits
3. **OAuth Error**: Ensure Google OAuth credentials are properly configured
4. **PDF Processing Error**: Check if PDF files are in the correct format and location

### Getting Help

- Check the logs in the terminal for detailed error messages
- Ensure all environment variables are properly set
- Verify all dependencies are installed correctly

## 🔮 Future Enhancements

- [ ] Support for more programming languages
- [ ] Interactive code execution
- [ ] Visual algorithm demonstrations
- [ ] Progress tracking and analytics
- [ ] Mobile app integration
- [ ] Multi-language support

---

**Built with ❤️ for DSA learners worldwide**