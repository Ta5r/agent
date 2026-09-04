# 🧠 The "Zero-AI-Jargon" Guide to RAG (Retrieval-Augmented Generation)
### *How to Search Documents by Meaning and Give Your AI an Open-Book Exam — Explained for Everyday Programmers*

---

## 🎯 Who Is This For?

You know how to write code: you know what a function, an array, a loop, a dictionary, a REST API, and a SQL database are. 

**However:**
- You have **never** taken an AI or Machine Learning course.
- Words like *"High-dimensional Vector Space"*, *"Embeddings"*, and *"Cosine Similarity"* sound like alien math.
- You want to look at [`rag.py`](file:///Users/tanay/mini-progs/agent/rag.py) and know **exactly** what every single line is doing, why it exists, and how it solves real software problems.

This manual explains RAG from scratch using standard programming concepts you already use every day.

---

## 📑 Table of Contents

1. [The Big Problem RAG Solves](#1-the-big-problem-rag-solves)
2. [What Does "RAG" Actually Stand For?](#2-what-does-rag-actually-stand-for)
3. [The Two "Obvious" Approaches That Fail](#3-the-two-obvious-approaches-that-fail)
4. [The Secret Sauce: What is an "Embedding"?](#4-the-secret-sauce-what-is-an-embedding)
5. [The Math in Plain English: What is "Cosine Similarity"?](#5-the-math-in-plain-english-what-is-cosine-similarity)
6. [The 4-Step RAG Pipeline (Mapped to `rag.py`)](#6-the-4-step-rag-pipeline-mapped-to-ragpy)
   - [Step 1: Chunking (The Sliding Window)](#step-1-chunking-the-sliding-window)
   - [Step 2: Vectorization & Caching](#step-2-vectorization--caching)
   - [Step 3: Searching (Top-K Matches)](#step-3-searching-top-k-matches)
   - [Step 4: Generation (Answering the User)](#step-4-generation-answering-the-user)
7. [Line-by-Line Code Breakdown of `rag.py`](#7-line-by-line-code-breakdown-of-ragpy)
8. [A Complete End-to-End Example](#8-a-complete-end-to-end-example)
9. [Programmer's Cheat Sheet](#9-programmers-cheat-sheet)

---

## 1. The Big Problem RAG Solves

Imagine you hire a brilliant university graduate. They know everything in public textbooks, grammar, world history, and general science.

Now you ask them:
> *"What is Quantum Tech Co.'s return policy for a damaged ErgoDesk?"*

They have **never seen** your internal company PDFs. What will happen?
1. They might admit: *"I don't know."*
2. Or worse, to sound helpful, they will **make up an answer** (*"You have 60 days and free shipping!"*). In AI terminology, this is called a **Hallucination**.

You can't retrain or fine-tune an AI model every time your legal team edits a PDF. Retraining takes weeks, costs thousands of dollars, and is overkill.

### The Solution: An Open-Book Exam
Instead of making the AI memorize your private documents, you give it an **open-book exam**:
1. When a user asks a question, your code searches your internal PDFs and finds the **1 or 2 paragraphs** that talk about that topic.
2. Your code pastes those 2 paragraphs into the prompt.
3. Your code tells the AI: *"Read these 2 paragraphs below and answer the user's question using ONLY this text."*

**That entire process is called RAG.**

---

## 2. What Does "RAG" Actually Stand For?

| Letter | Word | What It Means to a Programmer |
| :---: | :--- | :--- |
| **R** | **Retrieval** | Searching your local files (PDFs, docs, DB) to find snippets relevant to the user's question. |
| **A** | **Augmented** | Modifying/augmenting the input prompt by appending those retrieved snippets into it. |
| **G** | **Generation** | Letting the LLM generate a natural language response grounded in the retrieved facts. |

---

## 3. The Two "Obvious" Approaches That Fail

Before discovering RAG, every programmer tries one of two naive approaches. Let's see why both fail:

### Bad Idea #1: "Why not just paste the entire 200-page PDF into every prompt?"
- **Token Limits**: LLMs have a maximum input buffer (context window).
- **Cost**: Cloud AI providers charge per 1,000 tokens (words). Sending 200 pages on every single chat message will drain your API budget in hours.
- **Speed & Latency**: Sending 200 pages takes seconds to transfer and compute.
- **"Lost in the Middle"**: LLMs get confused when searching through massive walls of text for one tiny detail.

### Bad Idea #2: "Why not just use SQL `LIKE` or Regex `grep`?"
Suppose your PDF contains this rule:
> *"Reimbursement claims must be lodged within thirty days of delivery."*

Now, a customer asks:
> *"How long do I have to get my cash back?"*

Let's test standard keyword search:
- Does `"cash back"` appear in the PDF? **No.**
- Does `"how long"` appear in the PDF? **No.**
- Does `"return"` appear in that sentence? **No (it says 'reimbursement').**

**Keyword search returns 0 results.**
Human beings ask questions using synonyms, slang, typos, and indirect concepts. Computers need a way to match **meaning**, not exact characters.

---

## 4. The Secret Sauce: What is an "Embedding"?

This is where beginners get stuck. Let's completely demystify it.

### The GPS Analogy
How does Google Maps know that **San Francisco** and **Oakland** are close together, but **New York** is far away?
Because every city has **numerical coordinates**:
- San Francisco: `(37.77, -122.41)`
- Oakland: `(37.80, -122.27)` — *(Very close numbers!)*
- New York: `(40.71, -74.00)` — *(Very far numbers!)*

An **Embedding** does the exact same thing, but for **the meaning of words and sentences**.

### A 2D Toy Example
Imagine an AI that measures sentences on two simple axes:
- **Axis 1 (X)**: How much does this relate to **money/refunds**? (Scale: 0 to 10)
- **Axis 2 (Y)**: How much does this relate to **food/cooking**? (Scale: 0 to 10)

Let's plot three sentences:

```
 Food Axis (Y)
  ▲
10│               [S3: "Delicious pizza crust recipe"] (X=0.2, Y=9.5)
  │
 5│
  │
 0└───┬───────────────┬───────────────► Money/Refunds Axis (X)
     0               5               10
             [S1: "Reimbursement within 30 days"] (X=9.2, Y=0.1)
             [S2: "How do I get my cash back?"]  (X=9.0, Y=0.2)
```

Look at the coordinates:
- `S1`: `[9.2, 0.1]`
- `S2`: `[9.0, 0.2]`
- `S3`: `[0.2, 9.5]`

Even though `S1` and `S2` share **zero identical words**, their coordinates are almost on top of each other!
`S3` is way up on the food axis, far away from both.

### In the Real World: 768 Dimensions
Human language is too rich for just 2 axes. So Google's embedding model (`gemini-embedding-2`) evaluates text across **768 different semantic axes** simultaneously (tone, urgency, hardware, finance, action, geography, etc.).

When you pass a sentence to the embedding API:
```python
response = client.models.embed_content(
    model="gemini-embedding-2",
    contents="How do I get a refund?"
)
```
The model returns an ordinary Python array of 768 floating-point numbers:
```python
[-0.0182, 0.0451, -0.0093, 0.0821, ..., -0.0312]
```
**That array of numbers is the embedding.** Nothing more, nothing less.

---

## 5. The Math in Plain English: What is "Cosine Similarity"?

Once you have two coordinate arrays (vectors), how do you calculate how similar they are?

In high school geometry, you learned that two arrows pointing in the exact same direction have an angle of $0^\circ$. If they point in completely different directions, the angle is $90^\circ$.

**Cosine Similarity** calculates the cosine of the angle between two vectors:
- **`1.0`** $\implies$ The two vectors point in the exact same direction (**identical meaning**).
- **`0.0`** $\implies$ The two vectors are perpendicular (**completely unrelated**).
- **`-1.0`** $\implies$ The two vectors point in opposite directions.

### The Python Code in [`rag.py`](file:///Users/tanay/mini-progs/agent/rag.py#L144-L152):
```python
def cosine_similarity(v1, v2):
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    dot = np.dot(arr1, arr2)                 # Multiply matching coordinates and sum them up
    norm1 = np.linalg.norm(arr1)             # Calculate length of vector 1
    norm2 = np.linalg.norm(arr2)             # Calculate length of vector 2
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))      # Normalizes to a score between -1.0 and 1.0
```

You give it two lists of floats, and it gives you a similarity score (e.g. `0.88` = very strong match).

---

## 6. The 4-Step RAG Pipeline (Mapped to `rag.py`)

Here is the complete workflow of how our codebase handles documents and queries:

```
================================================================================
PHASE A: PREPARATION & INDEXING (Runs once or when files change)
================================================================================

 [ data/knowledge_base/refund_policy.pdf ]
                     │
                     ▼
 [Step 1: Chunking]  extract_pdf_chunks()
                     Slices PDF text into 500-character windows with 100-char overlap
                     │
                     ▼
 [Step 2: Embedding] client.models.embed_content(model="gemini-embedding-2")
                     Turns each 500-char chunk into a 768-float vector
                     │
                     ▼
 [Save to Cache]     data/rag_cache.json  (Saves vectors so we never re-pay for them)
                     _vector_index in memory


================================================================================
PHASE B: SEARCH & QUERY (Runs whenever customer asks a policy question)
================================================================================

 User Query: "Can I return an opened box?"
                     │
                     ▼
 [Step 3: Embed Query] Turns question into a 768-float vector
                     │
                     ▼
 [Cosine Similarity] Compares query vector against EVERY chunk vector in memory
                     Scores: [Chunk 1: 0.89, Chunk 2: 0.42, Chunk 3: 0.12]
                     │
                     ▼
 [Top-K Selection]   Picks the top 2 highest scoring chunks
                     │
                     ▼
 [Step 4: Answer]    Feeds top chunks to agent.py -> Gemini generates grounded answer!
================================================================================
```

---

### Step 1: Chunking (The Sliding Window)

A PDF might be 10 pages long. If we turn the whole 10 pages into a single embedding, the numbers get "watered down" by all the different topics on every page.

Instead, we slice the text into bite-sized **chunks** (e.g., 500 characters).

#### Why "Overlap"?
Imagine your PDF says:
> `"...orders cancelled before 5 PM get a full refund. Orders placed after..."`

If your chunker cuts strictly at character 500, it might split right in the middle:
- **Chunk 1**: `"...orders cancelled before 5 PM get a full"`
- **Chunk 2**: `"refund. Orders placed after..."`

Now neither chunk makes any sense!
To fix this, we use a **Sliding Window with Overlap**:
```
Chunk 1: [ Characters 0 to 500 ]
Chunk 2:       [ Characters 400 to 900 ]   <-- 100 characters repeated!
Chunk 3:             [ Characters 800 to 1300 ]
```
Because the last 100 characters of Chunk 1 are repeated at the beginning of Chunk 2, a sentence is never sliced in two without context.

---

### Step 2: Vectorization & Caching

Calling the Google API to turn text into embeddings costs a small fraction of a cent and takes ~100ms per chunk.
If you have 5 PDFs with 50 chunks each, you do **not** want to re-embed all 250 chunks every time your web server restarts!

In [`rag.py`](file:///Users/tanay/mini-progs/agent/rag.py#L74-L86), we use a smart caching pattern:
1. When inspecting a PDF, we check its file modification time (`os.path.getmtime(pdf_path)`).
2. If `data/rag_cache.json` already contains this filename and the timestamp matches:
   - **We load the vectors instantly from disk.** Zero API calls, zero latency!
3. If the timestamp changed (or it's a brand-new PDF):
   - We call `gemini-embedding-2`, get the vectors, and update `data/rag_cache.json`.

---

### Step 3: Searching (Top-K Matches)

When a customer asks:
> *"How many days do I have to return an item?"*

1. We embed that sentence: `query_vector = embed(query)`.
2. We loop through all chunks in our in-memory list `_vector_index`.
3. We run `cosine_similarity(query_vector, chunk["embedding"])`.
4. We sort the list from highest score to lowest.
5. We grab the top `K` items (in our code, `top_k = 2`).

---

### Step 4: Generation (Answering the User)

Now that we have the 2 best snippets from our PDF, [`agent.py`](file:///Users/tanay/mini-progs/agent/agent.py#L45-L53) formats them:
```
From policy document 'refund_policy.pdf' (Relevance: 0.89):
"Under Quantum Tech Co. policy, you can return items within 30 days of purchase in original, unused packaging..."
```
The AI reads this exact text and responds:
> *"Under our policy, you have 30 days from purchase to return items as long as they are in the original packaging."*

Notice: **The AI did not guess.** It cited the real policy directly from the PDF!

---

## 7. Line-by-Line Code Breakdown of `rag.py`

Let's look at the actual code in [`rag.py`](file:///Users/tanay/mini-progs/agent/rag.py) with simple explanations:

### 1. The Global In-Memory Store
```python
# Lines 15-16
# In-memory index of chunks: list of {"text": str, "source": str, "embedding": list}
_vector_index = []
```
* **What it is**: Just a plain Python list in RAM. Each element is a dictionary containing the chunk's text, the filename it came from, and its 768-float embedding array.

---

### 2. Extracting & Chunking PDF Text
```python
# Lines 27-46
def extract_pdf_chunks(pdf_path: str, chunk_size: int = 500, overlap: int = 100):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)  # Advance by 400 characters (leaving 100 overlap)
        
    return chunks
```
* **Line 29**: Uses the `pypdf` library to open the PDF.
* **Lines 31-34**: Loops over all pages and extracts raw text into one giant string `full_text`.
* **Lines 39-44**: The sliding window loop. If `chunk_size = 500` and `overlap = 100`, it moves forward by `400` characters on each step (`500 - 100 = 400`).

---

### 3. Smart Cache Verification
```python
# Lines 74-86
for filename in pdf_files:
    pdf_path = os.path.join(KB_DIR, filename)
    mtime = os.path.getmtime(pdf_path)
    
    # Check if cache is valid for this file
    if filename in cache and cache[filename].get("mtime") == mtime:
        print(f"RAG: Loading '{filename}' from cache...")
        for chunk_data in cache[filename].get("chunks", []):
            _vector_index.append({
                "text": chunk_data["text"],
                "source": filename,
                "embedding": chunk_data["embedding"]
            })
        continue
```
* **Line 75**: Gets the file's last modified timestamp (`mtime`).
* **Line 78**: If `filename` exists in `data/rag_cache.json` and the file hasn't been touched since then, load the embeddings straight into RAM and skip calling the API!

---

### 4. Calling the Embedding API
```python
# Lines 103-114
for chunk in chunks:
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk
    )
    embedding = response.embeddings[0].values
    embedded_chunks.append({
        "text": chunk,
        "embedding": embedding
    })
    time.sleep(0.1)  # Prevent hitting rate limits
```
* **Line 104**: Sends the 500-character chunk to Google.
* **Line 108**: Extracts the float array (`values`).
* **Line 114**: Pauses for 100ms between calls so we don't trigger HTTP 429 (Rate Limit Exceeded) errors.

---

### 5. Searching with Cosine Similarity
```python
# Lines 170-190
response = client.models.embed_content(
    model="gemini-embedding-2",
    contents=query
)
query_vector = response.embeddings[0].values

results = []
for item in _vector_index:
    sim = cosine_similarity(query_vector, item["embedding"])
    results.append({
        "text": item["text"],
        "source": item["source"],
        "score": sim
    })

# Sort by similarity score descending (highest match first)
results.sort(key=lambda x: x["score"], reverse=True)
return results[:top_k]
```
* **Lines 170-174**: Turns the customer's question into a 768-float vector.
* **Lines 180-186**: Calculates similarity between the question and every chunk in RAM.
* **Line 189**: Sorts results descending by score.
* **Line 190**: Returns only the top `top_k` (e.g. top 2) snippets.

---

## 8. A Complete End-to-End Example

Here is a real example showing what the data looks like at every stage of the pipeline:

### 1. Document in `data/knowledge_base/refund_policy.pdf`:
```
Section 4: Hardware Returns
Customers may return electronic items within 30 days of the purchase date. 
Items must be in original packaging. A 10% restocking fee applies to open boxes.
```

### 2. After Chunking & Embedding:
Stored in `_vector_index`:
```json
{
  "source": "refund_policy.pdf",
  "text": "Customers may return electronic items within 30 days of the purchase date...",
  "embedding": [0.038, -0.012, 0.089, ..., -0.045]
}
```

### 3. Customer Asks in Web Chat:
> *"Can I send back my projector if I already opened the box?"*

### 4. Vector Search Runs:
- Query vector: `[0.035, -0.010, 0.084, ..., -0.041]`
- Cosine similarity calculation:
  $$\text{Score} = 0.912 \quad (\text{Very high!})$$

### 5. Tool Output Given to Agent:
```
From policy document 'refund_policy.pdf' (Relevance: 0.91):
"Customers may return electronic items within 30 days of the purchase date. 
Items must be in original packaging. A 10% restocking fee applies to open boxes."
```

### 6. AI Generates Final Answer:
> *"Yes, you can return your projector within 30 days of purchase even if the box is opened, but please note that a 10% restocking fee will apply."*

---

## 9. Programmer's Cheat Sheet

Keep this table handy whenever you discuss RAG with other developers:

| Concept | What It Actually Is (In Plain English) | Where It Lives in Our Code |
| :--- | :--- | :--- |
| **Chunk** | A 500-character slice of text from a document | [`rag.py` (line 39)](file:///Users/tanay/mini-progs/agent/rag.py#L39) |
| **Overlap** | 100 characters shared between adjacent chunks to keep sentences whole | [`rag.py` (line 44)](file:///Users/tanay/mini-progs/agent/rag.py#L44) |
| **Embedding** | An array of 768 float numbers representing semantic meaning | [`rag.py` (line 108)](file:///Users/tanay/mini-progs/agent/rag.py#L108) |
| **Vector Index** | An in-memory Python list of dictionaries: `[{"text", "embedding"}]` | [`rag.py` (line 16)](file:///Users/tanay/mini-progs/agent/rag.py#L16) |
| **Cosine Similarity** | Normalized dot product scoring similarity from `-1.0` to `1.0` | [`rag.py` (line 144)](file:///Users/tanay/mini-progs/agent/rag.py#L144) |
| **Top-K** | Grabbing the top $K$ (e.g. 2) highest-scoring matching chunks | [`rag.py` (line 190)](file:///Users/tanay/mini-progs/agent/rag.py#L190) |
| **RAG Tool** | The Python function the AI calls when it needs policy facts | [`agent.py` (line 39)](file:///Users/tanay/mini-progs/agent/agent.py#L39) |

---

## 🏁 Summary

RAG is not magic. It is a 4-step engineering pattern:
1. **Slice text** into overlapping strings.
2. **Convert strings to float arrays** using an embedding API and cache them.
3. **Compare float arrays** using dot product (cosine similarity) to find the best match for a user's question.
4. **Feed the matching text** into the LLM prompt so it generates factually accurate answers.

Now you understand the entire codebase of [`rag.py`](file:///Users/tanay/mini-progs/agent/rag.py)!
