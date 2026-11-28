"""Day 4 - Tools, structured output and Google Sheets.

Sweet Crumbs agent that looks things up (get_price, is_open) and saves orders (log_order).
Also shows with_structured_output on messy messages and an agent trace.

Setup:
    pip install langchain langchain-groq pydantic
    pip install gspread            (only if you use Google Sheets)
    export GROQ_API_KEY="your_key"
Optional Google Sheets: share nothing publicly; create a service account, download its JSON key,
    export GOOGLE_SERVICE_ACCOUNT_FILE="path/to/key.json"
Without it, orders are saved to orders.csv next to this file.
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


# ---------- Structured output: messy message -> checked Order ----------
extractor = llm.with_structured_output(Order)

MESSY_MESSAGES = [
    "hi im ravi, need 2 choco cakes eggless for 24 dec pls",
    "Sravani here. one plum cake on saturday",
    "6 cupcakes for kiran, pickup 3rd jan, eggless ok",
    "this is anil, two chocolate cakes tomorrow evening, with egg is fine",
    "Meena - 12 cupcakes on 14 feb, no eggs please",
]


def structured_demo():
    for msg in MESSY_MESSAGES:
        try:
            print(msg, "->", extractor.invoke(msg))
        except Exception as e:
            print(msg, "-> could not extract:", e)


# ---------- The agent ----------
from langchain.agents import create_agent  # noqa: E402

SYSTEM = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Always use get_price for prices and is_open for timings; never guess them.
        Before log_order, make sure you have name, item, quantity and pickup date; ask if missing.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer."""

agent = create_agent(model=llm, tools=[get_price, is_open, log_order], system_prompt=SYSTEM)

history: list = []


def chat(text: str) -> str:
    """Send one message; keep the whole run (tool calls included) as memory."""
    global history
    try:
        result = agent.invoke({"messages": history + [{"role": "user", "content": text}]})
        history = result["messages"]
        return history[-1].content
    except Exception as e:
        print("Agent error:", e)
        return "Sorry, I'm a little busy. Please try again in a minute."


def run_tests(path: str = os.path.join(HERE, "test.csv")) -> float:
    """Each row is a fresh conversation: PASS if must_include is in the answer."""
    global history
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    passed = 0
    for row in rows:
        history = []
        answer = chat(row["question"])
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "->", answer[:80])
    history = []
    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")
    return passed / len(rows)


if __name__ == "__main__":
    print("--- Structured output on 5 messy messages ---")
    structured_demo()
    print("\n--- Agent ---")
    print(chat("Are you open on Sunday? How much are 2 chocolate cakes?"))
    print(chat("I'm Ravi. Please book 2 eggless chocolate cakes for 24 Dec."))
    print("\n--- Trace: question -> tool calls -> tool results -> answers ---")
    for m in history:
        m.pretty_print()
    print("\n--- Customer tests (donut, missing date, 0 cupcakes) ---")
    run_tests()
