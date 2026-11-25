# Day 2 - Prompts, memory and the Python agents need

## Task
Turn the Day-1 single call into a real chatbot for Sweet Crumbs Home Bakery that knows
its menu, remembers the customer and does not crash.

## What `task.py` does
- **Structured system prompt**: ROLE, FACTS, RULES, FORMAT.
- **Dict**: `MENU = {"plum cake": 500, ...}` is injected into the prompt with `json.dumps`.
- **Function with type hints + docstring**: `get_price(item, qty)`.
- **Decorator**: `@log_call` prints each call and its result.
- **Memory by hand**: `history` list is appended every turn and trimmed to the last
  `MAX_TURNS` messages before each request.
- **Error handling**: `RateLimitError`, `APIConnectionError` and `KeyError` handled with `try/except`.
- **JSON output**: `extract_order()` asks for name, item, qty, date, eggless and parses with `json.loads`.

## Run
```bash
pip install openai
export GROQ_API_KEY="your_key"
python task.py
```

## Key learnings
- Messages have roles: system (instructions), user (person), assistant (model reply).
- With `MAX_TURNS` too small, the bot forgets early details such as the customer's name.
- Asking for JSON and parsing it turns text into data a program can use.

## Before Day 3
Bring my 10 customer questions and my business catalogue as a Python dict.
