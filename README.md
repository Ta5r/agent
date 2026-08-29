# 🤖 Customer Support AI Agent & Developer Workspace

Welcome to the **Quantum Support Workspace**. This is an educational, interactive developer workspace demonstrating how to build your very first AI Agent. The agent is built using the new **Google GenAI SDK** (`google-genai`), integrated with a live **SQLite database** (simulating enterprise backend order lookup and ticket filing) and a local **PDF RAG (Retrieval-Augmented Generation)** knowledge base (simulating support policy documents).

The project features a **split-screen developer dashboard** that displays the inner workings of the agent as you chat with it.

---

## 🌟 Key Features

1. **Interactive Customer Chat UI**: A responsive, modern messaging interface simulating a customer interaction.
2. **Agent Reasoning Log**: A real-time terminal showing the agent's decision loop—including which tools the agent chose, the argument payloads, and execution outputs.
3. **SQLite Database Explorer**: A live view showing the state of database tables (`orders` and `tickets`), which flashes with a success animation when the agent files a new ticket.
4. **PDF RAG Knowledge Base Manager**: A drag-and-drop area to upload mock company policy PDFs. The system automatically reads, chunks, embeds (using `gemini-embedding-2`), and indices files into vector memory.
5. **Robust Guardrails**: Set instructions preventing the agent from answering out-of-scope inquiries (e.g. writing programming code or telling general jokes) or fabricating order details.

---

## 📂 Project Structure

```
agent/
├── main.py                 # FastAPI Web Server (Endpoints, serves static assets)
├── agent.py                # Gemini Agent logic, Tool definitions, & Guardrails loop
├── database.py             # SQLite setup, seeding, and database interfaces
├── rag.py                  # PDF parser, chunker, and semantic vector similarity search
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (contains GEMINI_API_KEY)
├── .gitignore              # Ignores .venv, database binaries, and private keys
├── data/
│   ├── support.db          # Auto-generated SQLite database
│   └── knowledge_base/     # Folder for RAG PDFs (seeded with refund_policy.pdf)
└── static/
    ├── index.html          # Chat interface and Developer Dashboard layout
    ├── style.css           # Glassmorphic styling variables and animations
    └── app.js              # State management, API calls, and live DB polling
```

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.10+** installed. Follow these setup steps:

### 1. Clone & Navigate to Workspace
```bash
git clone <repository-url>
cd agent/
```

### 2. Set Up Virtual Environment
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API Key
Create a `.env` file in the root directory (a template `.env` is provided) and paste your API key from [Google AI Studio](https://aistudio.google.com/):
```ini
GEMINI_API_KEY=AIzaSy...
```

---

## 🚀 How to Run the Project

With your virtual environment activated and `.env` configured, launch the server:

```bash
python main.py
```

Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)** to launch the developer workspace.

---

## 🧪 Testing Scenarios (Walkthrough)

Here are the scenarios you can use to test the workspace:

### Scenario 1: Order Status Lookup (Database Tool)
* **Message**: *"Can you check my order status?"*
* **Agent Response**: The agent will recognize it needs details and ask: *"I can check that. What is your Order ID and email address?"*
* **Message**: *"My ID is ORD-1002 and email is bob@example.com."*
* **Agent Response**: It queries SQLite and replies that the *UltraView 4K Projector* was **Shipped** with tracking code `TRK-102938`.
* **Developer Insight**: Open the **Agent Reasoning Log** tab on the right to see the JSON payload representing the database query executed by the agent.

### Scenario 2: Return Policy Verification (RAG Search)
* **Message**: *"How many days do I have to return an item?"*
* **Agent Response**: The agent queries the PDF index and replies: *"Under Quantum Tech Co. policy, you can return items within 30 days of purchase in original, unused packaging..."*
* **Developer Insight**: The agent read, extracted, and cited this directly from the seeded [`refund_policy.pdf`](file:///Users/tanay/mini-progs/agent/data/knowledge_base/refund_policy.pdf) file.

### Scenario 3: Filing a Ticket (Database Write)
* **Message**: *"My ErgoDesk (ORD-1003) is damaged. Can I get a refund? My email is charlie@example.com."*
* **Agent Response**: The agent looks up the order status and finds it is still "Processing". Recognizing the customer has a complaint, it will offer to file a support ticket.
* **Message**: *"Yes, file a ticket."*
* **Agent Response**: It registers a new support ticket and replies: *"Ticket ID 1 has been filed for you..."*
* **Developer Insight**: Open the **SQLite Database** tab. The new ticket immediately appears in the **Live Filed Tickets** table with a glowing green row animation.

### Scenario 4: Topic Guardrails
* **Message**: *"Write a Python script that calculates prime numbers."*
* **Agent Response**: The agent respects prompt constraints and replies: *"I am only authorized to assist with customer support inquiries for Quantum Tech Co. Let me know if you need help with an order, returns, or support tickets!"*

---

## 🔐 Security & Best Practices
The project includes a pre-configured `.gitignore` file that automatically prevents checking the following files into source control:
* `.env` (contains your private Gemini API Key)
* `data/support.db` (local SQLite database)
* `data/rag_cache.json` (cached embeddings)
* `.venv/` (virtual environment binaries)