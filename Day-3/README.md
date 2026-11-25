# Day 3 - Your first agent: what an agent is, and LangChain

## Task
Rebuild the Sweet Crumbs assistant with LangChain, give it memory, and test it with a CSV file.

## What `task.py` does
1. **LangChain chain**: `ChatPromptTemplate | ChatGroq | StrOutputParser`.
2. **Memory**: `MessagesPlaceholder("history")` plus an `ask()` function that stores each turn.
3. **Tests**: `run_tests()` reads `test.csv` (`question`, `must_include`), checks each answer
   and prints a pass rate.
4. **Agent sneak peek**: `create_agent(model, tools=[], system_prompt=SYSTEM)`.
   Tools will be added on Day 4.

## Run
```bash
pip install langchain langchain-groq
export GROQ_API_KEY="your_key"
python task.py
```

## Results
| Run | Prompt change | Temperature | Pass rate |
|-----|---------------|-------------|-----------|
| 1 | baseline | 0.3 | _fill in_ |
| 2 | _one change_ | 0.3 | _fill in_ |
| 3 | baseline | 1.0 | _fill in_ |
| 4 | baseline | 1.0 | _fill in_ |

## Agent or not? (my jobs)
| Job | Type (plain code / one LLM call / agent) |
|-----|------------------------------------------|
| e.g. Total price of an order | plain code |
| e.g. Reply to "do you make eggless cakes?" | one LLM call |
| e.g. Take an order, check price, log it, confirm | agent |

## Key learnings
- A chatbot talks; an agent decides steps and uses tools in a loop (think, act, observe).
- Known steps = plain code; one-off text = one LLM call; unknown steps + tools = agent.
- A test file with `must_include` makes prompt changes measurable.

## Before Day 4
Two business functions with type hints and docstrings (price lookup, availability check)
that will become the agent's tools.
