"""Day 3 - Your first agent: what an agent is, and LangChain.

Rebuilds the Sweet Crumbs assistant with LangChain (prompt | llm | parser), adds memory,
runs a CSV test suite with a pass rate, and shows a first create_agent with no tools.

Setup:
    pip install langchain langchain-groq
    export GROQ_API_KEY="your_key"
"""
import csv
import os

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

if "GROQ_API_KEY" not in os.environ:
    raise SystemExit("Set the GROQ_API_KEY environment variable first.")

TEMPERATURE = 0.3  # Switch task: try 1.0 and run the tests twice
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=TEMPERATURE)

SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80 each.
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time or place. If it isn't in FACTS,
        say "I'll check with the owner." Work out totals carefully.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),  # the chat so far goes here
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

history: list[tuple[str, str]] = []


def ask(question: str) -> str:
    """Ask with memory: the history list is passed on every call."""
    answer = chain.invoke({"history": history, "question": question})
    history.extend([("human", question), ("ai", answer)])
    return answer


def run_tests(path: str = os.path.join(os.path.dirname(__file__), "test.csv")) -> float:
    """Run every row of the CSV with no history; PASS if must_include is in the answer."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    passed = 0
    for row in rows:
        answer = chain.invoke({"history": [], "question": row["question"]})
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "->", answer[:70])
    rate = passed / len(rows)
    print(f"\n{passed}/{len(rows)} passed ({rate:.0%})")
    return rate


def agent_demo():
    """Sneak peek: same assistant as an agent. Tools arrive on Day 4."""
    agent = create_agent(model=llm, tools=[], system_prompt=SYSTEM)
    result = agent.invoke({"messages": [{"role": "user", "content": "Do you make eggless cakes?"}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    print("--- Memory demo ---")
    print(ask("Hi, I'm Ravi. Do you make eggless cakes?"))
    print(ask("What's my name?"))
    print("\n--- Tests ---")
    run_tests()
    print("\n--- Agent with empty tool list ---")
    agent_demo()
