"""Day 5 - Memory and RAG: an agent that remembers and looks things up.

Complete Sweet Crumbs single agent:
  - tools: get_price, is_open, log_order (Day 4) + search_faq (RAG over ChromaDB)
  - memory: a checkpointer with one thread_id per customer
  - tests: run_tests() with a fresh thread per question; aim for >= 80%

Setup:
    pip install langchain langchain-groq langgraph chromadb pydantic
    export GROQ_API_KEY="your_key"
Optional Google Sheets: see Day-4 (GOOGLE_SERVICE_ACCOUNT_FILE); else orders.csv is used.
"""
import csv
import os
from typing import Literal

from langchain.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

if "GROQ_API_KEY" not in os.environ:
    raise SystemExit("Set the GROQ_API_KEY environment variable first.")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_retries=3)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- The Orders "sheet" ----------
class LocalSheet:
    """Fallback when Google Sheets is not configured: appends rows to orders.csv."""

    def __init__(self, path: str):
        self.path = path

    def append_row(self, row: list) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


def open_orders_sheet():
    """Use Google Sheets if GOOGLE_SERVICE_ACCOUNT_FILE is set, else a local orders.csv."""
    header = ["name", "item", "qty", "date", "eggless", "total"]
    key_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if key_file:
        import gspread  # pip install gspread

        gc = gspread.service_account(filename=key_file)
        try:
            sh = gc.open("Sweet Crumbs Orders")
        except gspread.SpreadsheetNotFound:
            sh = gc.create("Sweet Crumbs Orders")
            sh.sheet1.append_row(header)
        print("Using Google Sheet:", sh.url)
        return sh.sheet1
    path = os.path.join(HERE, "orders.csv")
    if not os.path.exists(path):
        LocalSheet(path).append_row(header)
    print("Google Sheets not configured; saving orders to", path)
    return LocalSheet(path)


ws = open_orders_sheet()

# ---------- Data ----------
MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}
HOURS = {
    "monday": "9 am to 8 pm", "tuesday": "9 am to 8 pm", "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm", "friday": "9 am to 8 pm", "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm",
}


# ---------- Pydantic record: fields + types + rules ----------
class Order(BaseModel):
    """A cake order taken from a customer's chat message."""
    name: str = Field(description="Customer's first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item")
    qty: int = Field(description="How many", ge=1)
    date: str = Field(description="Pickup date as the customer said it")
    eggless: bool = Field(description="True if the customer asked for eggless")


# ---------- Tools ----------
@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants
    """
    price = MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"


@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. "sunday"."""
    hours = HOURS.get(day.lower().strip())
    return f"Open {hours} on {day}." if hours else f"'{day}' is not a day of the week."


@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Use only when the customer has given their name, item, quantity and pickup date."""
    try:
        order = Order(name=name, item=item.lower().strip(), qty=qty, date=date, eggless=eggless)
    except Exception as e:  # Pydantic rejects bad data before it reaches the sheet
        return f"Could not save the order: {e}. Ask the customer to correct it."
    total = MENU[order.item] * order.qty
    ws.append_row([order.name, order.item, order.qty, order.date,
                   "yes" if order.eggless else "no", total])
    return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."


# ---------- Knowledge: one fact per chunk in ChromaDB ----------
import chromadb  # noqa: E402

FAQS = {
    "delivery":     "We deliver within 5 km of the bakery. Delivery is Rs 50, free for orders above Rs 1,000.",
    "payment":      "We accept UPI and cash. Orders above Rs 1,000 need a 50% advance by UPI.",
    "cancellation": "Cancellation is free up to 12 hours before pickup. After that, the advance is not refunded.",
    "photo_cake":   "Photo cakes and custom designs need 48 hours' notice.",
    "notice":       "Regular orders need 24 hours' notice.",
    "eggless":      "All our cakes have eggless versions at the same price.",
    "allergens":    "Plum cake contains nuts and dried fruit. Cupcakes and chocolate cake are nut-free.",
    "storage":      "Keep cakes in the fridge and eat them within 3 days.",
    "message":      "A message on the cake is free, up to 20 letters.",
    "candles":      "Candles are free with every cake.",
    "discounts":    "Bulk-order discounts need the owner's approval.",
    "pickup":       "Pickup is during opening hours: Monday to Saturday 9 am to 8 pm, Sunday 10 am to 6 pm.",
}

chroma = chromadb.Client()  # in-memory database
faq = chroma.get_or_create_collection(name="sweet_crumbs_faq")
faq.upsert(ids=list(FAQS.keys()), documents=list(FAQS.values()))  # add or update


@tool
def search_faq(question: str) -> str:
    """Search Sweet Crumbs' policies and FAQs: delivery, payment, cancellation,
    photo cakes, allergens, storage, candles and messages on cakes.
    Use for any question about rules or policies, not for prices or opening hours."""
    results = faq.query(query_texts=[question], n_results=3)
    return "\n".join(results["documents"][0])


# ---------- The agent with memory ----------
from langchain.agents import create_agent  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

SYSTEM = """ROLE:   You are the friendly assistant for Sweet Crumbs Home Bakery.
RULES:  Use get_price for prices and is_open for timings; never guess them.
        For any policy question, use search_faq and answer ONLY from what it returns.
        If the answer isn't there, say "I'll check with the owner."
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""

agent = create_agent(
    model=llm,
    tools=[get_price, is_open, log_order, search_faq],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver(),
)


def chat(thread_id: str, text: str) -> str:
    """One thread_id per customer: each customer has their own conversation memory."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        return result["messages"][-1].content
    except Exception as e:
        print("Agent error:", e)
        return "Sorry, I'm a little busy. Please try again in a minute."


def run_tests(path: str = os.path.join(HERE, "test.csv")) -> float:
    """A fresh thread per question keeps the tests independent."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    passed = 0
    for i, row in enumerate(rows):
        answer = chat(f"test-{i}", row["question"])
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "->", answer[:80])
    rate = passed / len(rows)
    print(f"\n{passed}/{len(rows)} passed ({rate:.0%})")
    return rate


if __name__ == "__main__":
    print("RAG check:", faq.query(query_texts=["Can you send the cake to my house?"], n_results=1)["documents"][0])
    print(chat("ravi", "Hi, I'm Ravi. Do you deliver? My order is about Rs 1,300."))
    print(chat("ravi", "Does the plum cake have nuts?"))
    print(chat("sravani", "Can I cancel tomorrow morning if I pick up tomorrow evening?"))
    print(chat("ravi", "What's my name, and will I pay for delivery?"))  # memory + RAG together
    print(chat("sravani", "What's my name?"))  # different thread: must NOT know Ravi's name
    print("\n--- Tests ---")
    run_tests()
