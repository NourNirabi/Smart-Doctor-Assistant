# Smart Doctor Assistant 🩺🤖

A medical Large Language Model (LLM) system designed to assist individuals in understanding their symptoms and medical test results.

## 🧠 Overview

**Smart Doctor Assistant** is an AI-powered assistant that takes user-provided symptoms and analyzes them to identify possible diseases using Retrieval-Augmented Generation (RAG) integrated with **Ollama** and **ChromaDB**. The system offers personalized guidance and recommendations based on similar historical cases from reliable medical sources.

### ✨ Key Features

- Symptom analysis using local LLM with Ollama.
- Disease prediction powered by a custom ChromaDB vector database built from a dataset of 773 diseases.
- Uses Hybrid Retrieval (Dense + Sparse) for improved accuracy (Ollama Embeddings + BM25).
- Employs strict filtering to prevent guessing when no relevant match is found.
- Rejects vague or non-medical inputs politely.
- Simple and interactive UI built with Gradio.
- Reduces unnecessary doctor visits.
- Enhances public health awareness.

---

## 📚 Dataset

The dataset used for symptom-disease mapping was sourced from Kaggle:

🔗 [Diseases and Symptoms Dataset – Kaggle](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)

---

## 🛠️ Technologies Used

- [Ollama](https://ollama.com/)
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- BM25 (for keyword-based sparse retrieval)
- [Gradio](https://gradio.app/)
- [LangChain](https://www.langchain.com/)
  
---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/NourNirabi/Smart-Doctor-Assistant.git
cd Smart-Doctor-Assistant

```

2. **Install the required packages:**
   
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1.Navigate to the project directory:

```bash
cd src/models
```
2. Make sure you have Ollama installed and running locally.
3. Start the assistant interface:

```bash
python chatbot.py
```
A Gradio UI will open in your browser. Enter symptoms and get medical suggestions instantly.

## 📄 License
This project is licensed under the MIT License.

## 📬 Contact
For any questions or suggestions, feel free to contact:
📧 nournirabi@gmail.com

---
