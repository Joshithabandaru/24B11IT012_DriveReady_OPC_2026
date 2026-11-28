# Day 4 - Tools, structured output and Google Sheets

## Task
Upgrade the Sweet Crumbs chatbot into an agent that **looks things up and saves orders**.

## What `task.py` does
- **Tools** (`@tool`, snake_case names, type hints, docstrings saying when to use them):
  `get_price`, `is_open`, `log_order`.
- **Pydantic `Order` model**: fields, types and rules (`qty >= 1`, item must be on the menu).
  `log_order` builds an `Order(...)` itself, so bad data is rejected before it is saved.
- **Structured output**: `llm.with_structured_output(Order)` turns 5 messy customer messages into checked records.
- **Agent**: `create_agent(model, tools, system_prompt)` with chat history kept between turns.
- **Orders sheet**: `append_row` = one order, one row. Uses Google Sheets when
  `GOOGLE_SERVICE_ACCOUNT_FILE` is set, otherwise saves to `orders.csv`.
- **Trace**: `pretty_print()` shows question -> tool calls -> tool results -> answer.
- **Customer tests** in `test.csv`: a "donut" (not on the menu), an order without a date, and 0 cupcakes.
  After running, also check that no wrong rows appeared in the sheet / `orders.csv`.

## Run
```bash
pip install langchain langchain-groq pydantic
export GROQ_API_KEY="your_key"
python task.py
```
Optional Google Sheets: `pip install gspread`, create a service account, share nothing publicly,
and set `GOOGLE_SERVICE_ACCOUNT_FILE` to its JSON key (never commit that file).

## Key learnings
- Tools give the model live facts, actions and exact calculations; the model asks, my code runs.
- Good tools have clear names and docstrings; vague ones ("does stuff") are picked badly.
- Never fully trust model output that is about to be saved: validate with Pydantic.
- The `log_order` flow is check -> total -> save -> confirm, and only when the order is complete.

## Before Day 5
Write 15-20 facts about the business (policies, timings, delivery areas, FAQs).
