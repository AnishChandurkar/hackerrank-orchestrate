# Support Ticket Agent

A 3-stage pipeline that resolves support tickets automatically:

1. **Router** — rule-based safety layer that classifies and filters incoming tickets.
2. **Retriever** — BM25 search over the knowledge base to fetch relevant context.
3. **LLM** — Groq-hosted Llama model generates the final response using the retrieved context.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and set your Groq API key:

```bash
cp ../.env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can also export it directly in your shell:

```bash
# macOS / Linux
export GROQ_API_KEY=your_groq_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"
```

---

## Running the Agent

To validate against the sample tickets first:

```bash
python main.py --input ../support_tickets/sample_support_tickets.csv
```

```bash
python main.py
```

The agent reads tickets from `../support_tickets/support_tickets.csv` and writes
resolved responses to `../support_tickets/output.csv`.

---

## Architecture Decisions

- **BM25 over embeddings** — deterministic, zero-cost, no external vector DB required
- **Rule-based safety layer first** — catches injections and invalid tickets without burning API calls
- **One file = one chunk** — articles are self-contained, no splitting needed
- **Model** — `llama-3.3-70b-versatile` via Groq, `temperature=0`, `seed=42` for reproducibility
- **Confidence threshold** — auto-escalates if BM25 top score is below domain threshold

## Project Structure

```
code/
├── main.py          # Entry point — orchestrates the full pipeline
├── router.py        # Stage 1 — rule-based safety / ticket classification
├── retriever.py     # Stage 2 — BM25 retrieval over the knowledge base
├── llm.py           # Stage 3 — Groq/Llama API call for response generation
├── prompts.py       # Prompt templates used by the LLM stage
├── requirements.txt # Python dependencies
└── README.md        # This file
```
