# 🤖 AI Learning Assistant Chatbot

An AI-powered conversational chatbot built with **Python, LangChain, Groq, and Gradio**. The application allows users to ask questions about Python, Machine Learning, Deep Learning, and Data Science through a simple web-based chat interface.

## 🚀 Features

* **AI-Powered Responses:** Uses Groq's hosted `openai/gpt-oss-120b` model to generate answers.
* **Interactive Chat Interface:** Built with Gradio for browser-based conversations.
* **Conversation Memory:** Remembers previous messages within the same chat session.
* **Session Management:** Keeps conversations separate using unique session IDs.
* **Clear Chat:** Starts a fresh conversation without the previous session's history.
* **Download Chat:** Allows users to save their conversation as a text file.
* **Prompt Templates:** Uses LangChain chat prompt templates to guide the assistant's responses.
* **Environment Variables:** Loads the Groq API key securely from a `.env` file.

## 🛠️ Technologies Used

* Python
* LangChain
* LangChain Core
* LangChain Groq
* Groq API
* Gradio
* Python-dotenv

## 📂 Project Structure

```text
Rag/
│
├── app.py             # Gradio chatbot application
├── chatbot.py         # Original LangChain learning script (optional)
├── .env               # API key configuration
├── .gitignore         # Excludes secrets and unnecessary files
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the placeholder URL and folder name with your repository details.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install gradio langchain langchain-groq langchain-core langchain-community python-dotenv
```

### 4. Configure the API key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Get an API key from the Groq Console: https://console.groq.com/

Never publish your API key or commit your `.env` file to GitHub.

### 5. Run the application

```bash
python app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://127.0.0.1:7860
```

## 💬 How to Use

1. Launch the application.
2. Enter a question in the chat input.
3. Click **Send** or press Enter.
4. Continue asking follow-up questions to test conversation memory.
5. Click **Download Chat** to save the conversation.
6. Click **Clear Chat** to begin a new session.

## 🧠 Concepts Learned

This project demonstrates practical use of:

* Large Language Models (LLMs)
* LangChain model integration
* Chat prompt templates
* Message placeholders
* Conversation history and session IDs
* Runnable chains
* API key management
* Gradio user interfaces
* Python functions and event handling

## 🔮 Future Improvements

* Add PDF document question-answering using Retrieval-Augmented Generation (RAG).
* Store conversation history persistently.
* Add support for multiple AI models.
* Improve the interface with customizable chat settings.
* Deploy the chatbot on Hugging Face Spaces.
* Add error handling for API failures and rate limits.

## 🔐 Security

* Store API keys in environment variables.
* Exclude `.env` and virtual environment files from version control.
* Avoid committing personal or confidential conversation data.
* Use unique, safely managed files when deploying to multiple users.

## 👨‍💻 Author

**Abdul Rehman Hussain**

Aspiring AI / Machine Learning Engineer

GitHub: https://github.com/abdulrehmanhussain2007-blip

---

⭐ If you find this project useful, consider giving the repository a star.
