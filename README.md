📚 Genshin MiniRAG — Offline Assistant with Local LLM
A fully offline Retrieval-Augmented Generation (RAG) assistant for Genshin Impact.
Helps players answer questions about characters, builds, constellations, and more — no internet required.

🚀 Project Versions
✅ 1️⃣ Prototype
Initial proof of concept to run entirely locally on Android.

Uses a very lightweight local LLM and tiny dataset.

Shows that a mobile device can do local vector search + generation.

Focus: efficient, minimal RAM usage.

✅ 2️⃣ Local LLM — 2B Parameter Model
Upgraded to a 2 billion parameter open-weight model for better response quality.

Still 100% offline — no API calls to OpenAI or external services.

Suitable for more powerful Android devices or desktops.

Handles more complex, detailed queries.

✅ 3️⃣ Expanded Context Version
Adds a larger, richer dataset for retrieval.

Connects more in-depth context about characters, constellations, builds, and lore.

Uses local embeddings for improved matching of user queries to the right text chunks.

Balances bigger context with local device resource limits.

📁 Datasets
📊 Sources:

Genshin Impact Characters Dataset
→ Kaggle dataset providing structured info about all characters, elements, weapons, stats, etc.

genshin_dataset_cleaned.jsonl
→ A cleaned version of transcriptions from various YouTubers’ build guide videos.
→ Provides real community strategies and tips to help the LLM give practical, contextual answers.

🔍 Local embedding store:
Uses these datasets for retrieval before passing the context to the local LLM.

🔑 How It Works
MiniRAG: Combines local vector database + local LLM.

No cloud required: All embeddings and models are local.

Flexible: You can expand the dataset anytime with more game updates.

Optimized: Prototype is tiny; 2B+ versions need a stronger GPU or device.

🗂️ Repo Structure
prototype/ — Mobile-first POC.

local_llm/ — 2B model version.

expanded/ — Extended dataset version with both Kaggle and community guides.

⚡ Features
✅ Character-specific answers: weapons, builds, constellations.

✅ Works fully offline.

✅ Local vector search ensures relevant context.

✅ Cross-platform: Linux, Windows, Android (powerful enough hardware).

📌 Roadmap
Add more up-to-date constellations data.

Experiment with different quantized models for lower-end phones.

Bundle everything as an APK for easy install.

Add a simple chat interface for mobile.

📑 Credits
Prototype, expansions, and local LLM setup by Akki Maharaj

Datasets:

Kaggle — Genshin Impact Characters

genshin_dataset_cleaned.jsonl — YouTube transcriptions (cleaned and community-friendly).

🔗 Repo
Akki-Maharaj/Genshin-MiniRAG-Offline-Assistant-with-Local-LLM
