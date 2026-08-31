from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3.5:2b"

# This is Abstraction Layer 1 - FUNCTION CALLING
# NO LangChain objects allowed
# will use tools from the OLLAMA library instead


# --- Tools ---
# without LangChain, functions need conversion to OLLAMA tools in accordance with OLLAMA specs (if using OLLAMA, different specs for using the ANTHROPIC or OPENAI sdks)
# most importantly: manually define a JSON schema for each function must include Google-style docstrings (@tool does this automatically)


@traceable(run_type = 'tool') # @tool implicitly tracks, but can also be done explicitly like this
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type = 'tool')
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# JSON schema
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
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
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

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# without LangChain, LLM calls must be manually traced for LangSmith
@traceable(name = 'Ollama Chat', run_type = 'llm')
def ollama_chat_traced(messages):
    return ollama.chat(model = MODEL, tools = tools_for_llm, messages = messages)


# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # declare which tools will the LLM have access to
    tools_dict = {
        'get_product_price': get_product_price,
        'apply_discount': apply_discount,
    } 

    print(f"Question: {question}")
    print("=" * 60)

    ### REACT DIAGRAM ELEMENT: the Query
    # without LangChain there are no natively defined message types, they need explicit defining
    # the following structure is OLLAMA specific, other providers' may vary
    messages = [
        {
            'role': 'system',
            'content': (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
                "5. If the user does not specify a particular type or model of an item"
                "from the catalog, just pick a generic one, do NOT ask the user for"
                "specific name, type or model."
            ),
        },
        {'role': 'user', 'content': question},
    ]
    ### REACT DIAGRAM ELEMENT: the Query

    ### REACT DIAGRAM ELEMENT: the Agent Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ### REACT DIAGRAM ELEMENT: the Thought
        # with LangChain, can't use the convenient .invoke
        response = ollama_chat_traced(messages = messages)
        ai_message = response.message # this is an OLLAMA response object, different from Langchain's
        ### REACT DIAGRAM ELEMENT: the Thought

        ### REACT DIAGRAM ELEMENT: the Action
        tool_calls = ai_message.tool_calls
        # If no tool calls, this is the final answer since no tools = LLM has all it needs for final answer
        if not tool_calls:
            ### REACT DIAGRAM ELEMENT: the Answer
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
            ### REACT DIAGRAM ELEMENT: the Answer
        ### REACT DIAGRAM ELEMENT: the Action

        ### REACT DIAGRAM ELEMENT: the Tool
        # LLM might consider multiple tools, but we'll process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name # can't use the dict anymore
        tool_args = tool_call.function.arguments
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        # check if name of tool chosen by LLM actually exists
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # invoke the tool runnable
        observation = tool_to_use(**tool_args) # no .invoke without LangChain
        ### REACT DIAGRAM ELEMENT: the Tool

        ### REACT DIAGRAM ELEMENT: the Observation
        print(f"  [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
                'role':'tool',
                'content': str(observation), 
            }
        ) 
        ### REACT DIAGRAM ELEMENT: the Observation

    print("ERROR: Max iterations reached without a final answer")
    return None
    ### REACT DIAGRAM ELEMENT: the Agent Loop


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")