from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "gpt-oss"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print("get_product_price")
    price = {"laptop": 1299.99, "headphones": 149.95}
    return price.get(product, 0)

@tool
def apply_discount(pice: float, discount_tier: str) -> float:
    """Apply a discount to a pice.
    Available tiers: bronze, silver, gold."""
    print("apply_discount")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(pice * (1 - discount/100), 2)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print("run_agent is running...")

    messages = [
        SystemMessage(
            content=(
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
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print("run_agent iteration:", iteration)

        ai_messages = llm_with_tools.invoke(messages)

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
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")