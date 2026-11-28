# Day 5 - Memory and RAG: an agent that remembers and looks things up

## Task
Finish the complete single Sweet Crumbs agent: tools + memory + knowledge + tests.
Also build a small RAG system over a PDF.

## Part 1 - `task.py` (Sweet Crumbs agent)
- **Knowledge (RAG)**: 12 bakery FAQs stored in a ChromaDB collection, one fact per chunk.
- **`search_faq` tool**: searches by meaning, so "Can you send the cake to my house?" finds the delivery fact.
- **Memory**: `InMemorySaver` checkpointer, one `thread_id` per customer. A second thread does not
  know the first customer's name.
- **Rules in the prompt**: answer policy questions only from `search_faq`; otherwise say
  "I'll check with the owner."
- **Tests**: `test.csv` has the Day 3 questions plus the Day 5 ones; `run_tests()` uses a fresh thread
  per question and prints the pass rate (target: 80% or more).

```bash
pip install langchain langchain-groq langgraph chromadb pydantic
export GROQ_API_KEY="your_key"
python task.py
```

## Part 2 - `rag_pdf.py` (RAG over `The_Little_Lantern.pdf`)
Pipeline: read PDF (OCR fallback for scanned pages) -> chunk (500 characters, 100 overlap) ->
embed with `all-MiniLM-L6-v2` (384 dimensions) -> FAISS index -> retrieve top 3 chunks ->
a Groq model answers **only** from that context, or says "The answer is not available in the PDF."

```bash
pip install pymupdf sentence-transformers faiss-cpu numpy groq
export GROQ_API_KEY="your_key"
python rag_pdf.py
```

**Sample questions to try:** What is the name of the boy? Why did Arjun place the lantern outside?
Who is Koh? (not in the PDF, so it should refuse). In my first run, the question "with whom Arjun
lived with?" was answered "not available" even though the PDF says he lived with his grandmother;
phrasing and chunking affect retrieval, which is a good example of RAG's limits.

## Key learnings
- Three kinds of memory: conversation (threads), customer (the sheet), knowledge (the vector database).
- Do not put everything in the prompt: search first, send only what is relevant.
- Embeddings turn text into numbers; similar meaning means nearby numbers.
- One idea per chunk, with a small overlap, keeps retrieval accurate.
- RAG = retrieve -> augment -> generate, and "not found" is an acceptable answer.

## Test results
| Run | Pass rate |
|-----|-----------|
| 1 | _fill in after running_ |

## Before Day 6
Draw on paper the steps my business follows for one customer request.
