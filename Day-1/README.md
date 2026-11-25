# Day 1 - How LLMs work and your first API call

## Task
Make a first API call to an LLM through Groq and explore how it behaves, using the
Sweet Crumbs Home Bakery case study.

## What `task.py` does
| # | Experiment | What to observe |
|---|-----------|-----------------|
| 1 | System prompt (role, facts, rules, format) | The bot follows the bakery's rules |
| 2 | Temperature 0.2 vs 1.2, same prompt 3 times each | Low = consistent, high = varied |
| 3 | Remove FACTS, ask "Do you deliver to Kakinada?" | Without facts the model may invent an answer |
| 4 | Two separate calls vs one messages list | The model has no memory; code must resend history |

## Run
```bash
pip install langchain-groq
export GROQ_API_KEY="your_key"
python task.py
```

## Key learnings
- An LLM predicts the next token; it does not "know" your business.
- Facts and rules in the system prompt reduce hallucination.
- Temperature controls creativity vs consistency.
- Memory is not built in: every call must include the earlier messages.

## Before Day 2
Write down the 10 questions customers ask my business most.
