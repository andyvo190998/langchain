from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gpt-oss"

import ollama


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print("get_product_price")
    price = {"laptop": 1299.99, "headphones": 149.95}
    return price.get(product, 0)


@traceable(run_type="tool")
def apply_discount(pice: float, discount_tier: str) -> float:
    """Apply a discount to a pice.
    Available tiers: bronze, silver, gold."""
    print("apply_discount")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(pice * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print("run_agent is running...")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "you MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount after you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "always use the apply_count tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one"
            )
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print("run_agent iteration:", iteration)

        responses = ollama_chat_traced(messages)
        ai_messages = responses.messages
        tool_calls = ai_messages.tool_calls

        if not tool_calls:
            print(f"\nFinal answer: {ai_messages.content}")
            return ai_messages.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_arguments = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool selected]: {tool_name} with arguments: {tool_arguments}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError("Tool error")

        observation = tool_to_use.invoke(tool_arguments)
        print(f" [Tool used]: {tool_name} with observation: {observation}")

        messages.append(ai_messages)
        messages.append(
            {
                "role": "rool",
                "content": str(observation),
            }
        )
        print("ERROR: Max iterations reached without a final answer")
        return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")
