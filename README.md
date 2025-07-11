# ✨ Genshin MiniRAG: Offline Assistant with Local LLM

An offline Retrieval-Augmented Generation (RAG) pipeline for a Genshin Impact assistant using a lightweight local LLM — no internet needed!

---

## 🎮 Project Overview

This project is a proof-of-concept **Genshin Impact helper bot** that runs entirely offline.  
It combines a local knowledge base with semantic search and a small LLM to answer player queries about characters, builds, or lore.

The goal is to make an efficient, privacy-friendly assistant for players, especially on low-resource devices.

---

## 📂 Key Features

✅ Lightweight **RAG pipeline** (Retrieval-Augmented Generation)  
✅ Uses **local embeddings** to find relevant context chunks  
✅ Generates answers using a **small local LLM**  
✅ Runs fully offline, no internet or cloud calls required  
✅ Optimized for minimal RAM and CPU usage

---

## 📚 Dataset

- A custom **JSONL file** built from cleaned YouTube transcriptions about Genshin characters, builds, and gameplay tips.
- Text chunks are embedded for semantic search to find the most relevant context for each question.

---

## 🛠️ Technologies Used

- **Python**
- **LangChain**
- **FAISS** or **Chroma** (vector store for embeddings)
- **Local LLM** (e.g., `Phi-2` or similar)
- **SentenceTransformers**
- **Streamlit** (optional UI)

---

## 📌 Project Structure

```plaintext
Genshin-MiniRAG-Offline-Assistant-with-Local-LLM/
├── data/               # JSONL knowledge base
├── embeddings/         # Saved vector embeddings
├── app.py              # Main RAG pipeline and chat interface
├── models/             # Local LLM weights or configs
├── requirements.txt    # Project dependencies
└── README.md
```


---

## ⚙️ How It Works

1. **Preprocess** the dataset into clean text chunks.
2. **Embed** the chunks using a sentence embedding model.
3. Store embeddings in a local vector store (**FAISS**, **Chroma**, etc.).
4. For each query, run **semantic search** to find relevant context.
5. Pass the context and query to the **local LLM** to generate the final answer.
6. Optionally run a simple **Streamlit** UI for local chat.

---

## 📈 Future Improvements
Add support for larger local LLMs with quantization for better answers.

Improve chunking and retrieval accuracy.

Build a mobile-friendly offline app.

## 🙌 Acknowledgements
Inspired by open-source RAG examples using LangChain and local LLMs.

Genshin Impact knowledge sourced from publicly available guides and transcriptions.

## 📬 Contact
- GitHub: Akki-Maharaj
- Linkedin: https://www.linkedin.com/in/akshat--/
Feel free to open issues or suggest improvements!

---
