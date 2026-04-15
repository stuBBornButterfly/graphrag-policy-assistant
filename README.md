# 🏛️ Bangladesh Education Policy — GraphRAG Multi-Agent Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> A production-grade **GraphRAG + Multi-Agent AI system** that answers complex questions
> about policy documents using 4 specialized AI agents, a knowledge graph, and semantic search —
> far beyond what a basic RAG chatbot can do.

---

## 🚀 Live Demo

👉 **[Try the live app here](https://graphrag-policy-assistant-lrfplyjwveelyr22ysu7qd.streamlit.app/)**

Enter your free [Groq API key](https://console.groq.com) in the sidebar and start asking questions.

---

## 📌 The Problem This Solves

Traditional RAG chatbots:
- ❌ Search by keywords only
- ❌ Look at one document at a time
- ❌ Miss contradictions between documents
- ❌ Cannot trace cause-and-effect chains
- ❌ Can hallucinate facts

This system:
- ✅ Searches by **meaning** (semantic similarity)
- ✅ Connects knowledge **across all documents**
- ✅ Detects **contradictions** between policies
- ✅ Traces **cause → effect** chains
- ✅ **Verifies every claim** before output

---

## 🧠 Architecture

```
                        User Question
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       🕐 TimeAgent   ⚠️ ContradictionAgent  💡 ImpactAgent
       (timeline)       (conflicts)         (cause-effect)
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                   🛡️ GroundingAgent
                   (fact verification)
                              │
                              ▼
                  📊 Final Synthesized Answer
```

### How Each Agent Works

| Agent | Job | Technique |
|---|---|---|
| 🕐 **TimeAgent** | Tracks how policies changed across years | Temporal semantic retrieval |
| ⚠️ **ContradictionAgent** | Finds conflicts between documents | Cross-document comparison |
| 💡 **ImpactAgent** | Traces cause → effect chains | Graph traversal + RAG |
| 🛡️ **GroundingAgent** | Verifies every claim against sources | Evidence-based validation |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | LLaMA 3.1 8B (Groq API) | Agent reasoning |
| **Vector Database** | ChromaDB | Semantic chunk retrieval |
| **Embeddings** | all-MiniLM-L6-v2 | Text → vector conversion |
| **Knowledge Graph** | NetworkX | Entity relationship mapping |
| **Frontend** | Streamlit | User interface |
| **Deployment** | Streamlit Cloud | Free live hosting |
| **Development** | Kaggle Notebook | GPU + free environment |

---

## ✨ Key Features

- 🔍 **Semantic Search** — Retrieves chunks by meaning, not keywords
- 🕸️ **Knowledge Graph** — Maps entities (programs, budgets, organizations) and their relationships across all 5 policy documents
- 🤖 **4 Specialist Agents** — Each has a single focused job for better accuracy
- 🛡️ **Anti-Hallucination Layer** — Claims marked ✅ VERIFIED, ❌ UNVERIFIED, or ⚠️ PARTIAL before surfacing
- ⚡ **Real-time Inference** — LLaMA 3.1 on Groq delivers responses in under 3 seconds
- 🌐 **Fully Deployed** — Live public URL, shareable portfolio link

---

## 📁 Project Structure

```
graphrag-policy-assistant/
├── app.py                  # Full Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/stuBBornButterfly/graphrag-policy-assistant.git
cd graphrag-policy-assistant
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

**Step 4 — Add your API key**

Get a free Groq API key at [console.groq.com](https://console.groq.com) and enter it in the sidebar.

---

## 💬 Example Questions

### Timeline Questions (tests TimeAgent)
- *"How did girls enrollment change between 2015 and 2024?"*
- *"What happened to the dropout rate over the years?"*
- *"How did budget allocations grow from 2015 to 2024?"*

### Contradiction Questions (tests ContradictionAgent)
- *"Are there any conflicts between budget decisions and enrollment targets?"*
- *"Did the Finance Ministry and Education Ministry ever disagree?"*
- *"Was the 2019 budget consistent with education goals?"*

### Impact Questions (tests ImpactAgent)
- *"What was the impact of the Girls Stipend Program on literacy rates?"*
- *"How did the World Bank loan affect education outcomes?"*
- *"What caused the 31% enrollment increase by 2022?"*

### Stress Test (tests GroundingAgent)
- *"Did girls enrollment reach 50% in engineering programs by 2020?"*
  > ← This should return ❌ UNVERIFIED — proving the anti-hallucination layer works!

---

## 📊 Results

| Question Type | Basic RAG | This System |
|---|---|---|
| Timeline across 5 docs | ❌ Misses context | ✅ Full timeline |
| Contradiction detection | ❌ Cannot detect | ✅ Flags conflicts |
| Cause-effect tracing | ❌ Single doc only | ✅ Cross-doc chains |
| Hallucination check | ❌ No verification | ✅ Every claim verified |

---

## 🗺️ Roadmap

- [ ] Real PDF upload support
- [ ] Knowledge graph visualization in the UI
- [ ] Support for any country's policy documents
- [ ] LangGraph-based agent orchestration
- [ ] Add memory across sessions

---

## 👤 Author

**Ayesha Siddika**
Final Year Student | AI · ML · NLP · Blockchain

- 🐙 GitHub: [@stuBBornButterfly](https://github.com/stuBBornButterfly)
- 💼 LinkedIn: [Ayesha Siddika](https://www.linkedin.com/in/ayesha-siddika-960140294/)
---

