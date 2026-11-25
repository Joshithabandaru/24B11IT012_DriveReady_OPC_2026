"""Day 1 - How LLMs work and your first API call.

Business case: Sweet Crumbs Home Bakery.
Runs the four Day-1 experiments: system prompt, temperature, hallucination, no memory.

Setup:
    pip install langchain-groq
    export GROQ_API_KEY="your_key"      (Windows: set GROQ_API_KEY=your_key)
"""
import os
from langchain_groq import ChatGroq

if "GROQ_API_KEY" not in os.environ:
    raise SystemExit("Set the GROQ_API_KEY environment variable first.")

MODEL = "openai/gpt-oss-120b"

SYSTEM = """You are the assistant for Sweet Crumbs Home Bakery.
Facts: we bake to order; orders need 24 hours' notice; all cakes have eggless versions.
Rules: never promise delivery times; if you do not know, say so.
Format: reply politely in under 60 words."""


def make_llm(temperature: float = 0) -> ChatGroq:
    return ChatGroq(model=MODEL, temperature=temperature)


def ask(question: str, system: str = SYSTEM, temperature: float = 0) -> str:
    """One stateless call: system prompt + one user question."""
    return make_llm(temperature).invoke([("system", system), ("human", question)]).content


def experiment_system_prompt():
    print("\n=== 1. First call with a system prompt ===")
    print(ask("can i get my order tommorrow"))


def experiment_temperature():
    print("\n=== 2. Temperature: tagline x3 at 0.2, x3 at 1.2 ===")
    for temp in (0.2, 1.2):
        print(f"-- temperature {temp}")
        for _ in range(3):
            print("  ", ask("Write a tagline for Sweet Crumbs", temperature=temp))


def experiment_hallucination():
    print("\n=== 3. Hallucination: no FACTS, ask about Kakinada delivery ===")
    print(ask("Do you deliver to Kakinada?", system="You are a bakery assistant."))
    print("-- with FACTS and rules:")
    print(ask("Do you deliver to Kakinada?"))


def experiment_no_memory():
    print("\n=== 4. No memory: separate calls vs one messages list ===")
    llm = make_llm()
    print("Call 1:", llm.invoke([("human", "My name is Ravi")]).content)
    print("Call 2 (separate):", llm.invoke([("human", "What is my name?")]).content)
    together = [
        ("human", "My name is Ravi"),
        ("ai", "Nice to meet you, Ravi!"),
        ("human", "What is my name?"),
    ]
    print("Together:", llm.invoke(together).content)


if __name__ == "__main__":
    experiment_system_prompt()
    experiment_temperature()
    experiment_hallucination()
    experiment_no_memory()
