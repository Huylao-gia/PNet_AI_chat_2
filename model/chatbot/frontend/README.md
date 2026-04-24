
# 🐾 Pet Care Chatbot - Streamlit Frontend

This directory contains the lightweight, interactive Streamlit frontend for the Pet Care RAG Chatbot. It connects directly to the FastAPI backend and provides a ChatGPT-like user experience with real-time text streaming.

## ✨ Features

-   **Real-time Streaming:** Consumes Server-Sent Events (SSE) from the backend API to display AI responses token-by-token.
    
-   **Session Memory:** Automatically generates and maintains a unique `session_id` per browser tab, ensuring the AI remembers conversational context (History).
    
-   **Dynamic Configuration:** A sidebar allows users to easily point to different backend URLs and adjust the `Top K` document retrieval parameter on the fly.
    
-   **Chat Management:** Built-in functionality to clear chat history and reset the session context with a single click.
    

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.10+
    
-   The FastAPI Backend server must be running and accessible (default is `http://localhost:8080`).
    

### 2. Installation

Navigate to the `frontend/` directory and install the required dependencies:

```
pip install streamlit requests

```

### 3. Running the Application

Launch the Streamlit app using the following command:

```
streamlit run app.py

```

This will start a local web server and automatically open the application in your default web browser (usually at `http://localhost:8501`).

## 🛠️ Usage Guide

1.  **Verify Backend Connection:** Ensure the `Backend API URL` in the left sidebar points to your active FastAPI backend endpoint.
    
2.  **Adjust Top K (Optional):** Use the slider in the sidebar to define how many context documents the VectorDB should retrieve per question. Higher values provide more context to the AI but consume more tokens.
    
3.  **Start Chatting:** Type your pet-related questions into the chat input box at the bottom of the screen.
    
4.  **Clear History:** Click the "🗑️ Xóa lịch sử chat" button in the sidebar to wipe the screen and start a fresh session (this generates a new `session_id`, isolating it from previous chats).
    

## 📁 File Structure

-   `app.py`: The main Streamlit application script containing the UI layout, state management, and SSE streaming logic.
    
-   `README.md`: Documentation for the frontend module.
