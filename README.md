# MCP Expert Chatbot

This project is a sophisticated, command-line-based chatbot designed to answer questions about the Model Context Protocol (MCP). It leverages a Retrieval-Augmented Generation (RAG) architecture, using Google's Gemini Pro model combined with a local knowledge base to provide accurate and contextually relevant answers.

## Features

- **Specialized Knowledge**: Expertly trained on the Model Context Protocol (MCP).
- **RAG Architecture**: Uses a local ChromaDB vector store to retrieve relevant documents, which are then fed to the language model for generating informed responses.
- **Command-Line Interface**: A simple and intuitive CLI for direct interaction with the chatbot.
- **Automated Setup**: Comes with a `start.sh` script that handles virtual environment creation, dependency installation, and configuration checks.
- **Configurable**: Easily configured via a `.env` file for your API key and other settings.
- **Extensible Knowledge Base**: The knowledge base can be rebuilt or expanded to include new information about MCP.

## System Architecture

The chatbot operates on a simple yet powerful architecture:

1.  **User Input**: The user enters a query through the CLI.
2.  **Embedding & Similarity Search**: The user's query is converted into an embedding. This embedding is used to search the ChromaDB vector store for the most similar and relevant documents from the knowledge base.
3.  **Prompt Construction**: The retrieved documents are combined with the original user query and a system prompt to create a comprehensive context for the language model.
4.  **LLM Response Generation**: The constructed prompt is sent to the Gemini API.
5.  **Display Response**: The generated response from the model is displayed to the user in the CLI.

## Prerequisites

- Python 3.8 or higher
- `git` for cloning the repository

## Getting Started

Follow these steps to get the MCP Expert Chatbot running on your local machine.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Q-A-Chatbot-About-MCP-Servers
```

### 2. Configure Your API Key

The chatbot requires a Google Gemini API key to function.

1.  **Create a `.env` file**:
    The project includes a startup script that will automatically create a `.env` file for you if one doesn't exist. Alternatively, you can create it manually:

    ```bash
    cp .env.example .env
    ```

2.  **Add your API Key**:
    Open the `.env` file and replace `your_api_key_here` with your actual Gemini API key.
    ```bash
    GEMINI_API_KEY=your_actual_gemini_api_key
    ```
    You can obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the Chatbot

The easiest way to start the chatbot is by using the provided shell script. It will automatically set up a virtual environment, install all dependencies, and launch the application.

```bash
bash start.sh
```

Once the script finishes, you will be greeted by the chatbot prompt directly in your terminal.

## Usage

After launching the chatbot, you can interact with it by typing your questions and pressing Enter.

```
🤖 MCP Expert Chatbot (CLI Mode)
Type 'exit' or 'quit' to end the chat.
----------------------------------------
You: What is MCP?
```

To end your session, simply type `exit` or `quit`.

---

_This project was developed with the assistance of an AI pair programmer._
