"""Day 2 - Prompts, memory and the Python agents need.

Builds the Sweet Crumbs chatbot: structured system prompt, menu dict, a documented
function, hand-made memory with trimming, error handling, and JSON order extraction.

Setup:
    pip install openai
    export GROQ_API_KEY="your_key"
"""
import functools
import json
import os

from openai import APIConnectionError, OpenAI, RateLimitError

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 8  # keep only the most recent messages (plus the system prompt)

# Dict: the menu (key -> value)
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}

SYSTEM = f"""ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
FACTS:  Menu with prices in rupees: {json.dumps(MENU)}
        Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never promise a delivery time. If it isn't in FACTS, say "I'll check with the owner."
        Remember the customer's name and order during the chat.
FORMAT: Under 50 words, friendly, English or Telugu matching the customer."""


# Decorator: logs every call to the wrapped function
def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [log] {func.__name__}{args} -> {result!r}")
        return result
    return wrapper


@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the menu."""
    return MENU[item.lower().strip()] * qty


# Memory by hand: append, send all (trimmed), store the reply
history = [{"role": "system", "content": SYSTEM}]


def chat(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    recent = [history[0]] + history[1:][-MAX_TURNS:]
    try:
        response = client.chat.completions.create(model=MODEL, messages=recent, temperature=0.3)
        reply = response.choices[0].message.content
    except RateLimitError:
        reply = "We're a little busy. Please try again in a minute."
    except APIConnectionError:
        reply = "I can't reach the server right now. Please try again shortly."
    history.append({"role": "assistant", "content": reply})
    return reply


def extract_order(text: str) -> dict | None:
    """Ask the model for an order as JSON, then json.loads it safely."""
    prompt = (
        "Return ONLY JSON with keys name, item, qty, date, eggless (true/false). "
        f"Order message: {text}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Model did not return valid JSON:", raw)
        return None


def demo():
    print("get_price demo:", get_price("cupcake", 6))  # 480
    try:
        get_price("donut")
    except KeyError:
        print("donut is not on the menu")  # error handling with try/except
    print("order JSON:", extract_order("Hi, I'm Ravi, 2 eggless plum cakes for 30 Nov"))


if __name__ == "__main__":
    demo()
    print("\nChat with Sweet Crumbs (type bye to quit)")
    while True:
        text = input("You: ")
        if text.lower().strip() in ("bye", "exit", "quit"):
            print("Bot: Thank you for visiting Sweet Crumbs!")
            break
        print("Bot:", chat(text))
