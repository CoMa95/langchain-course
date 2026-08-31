from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

import re # needed to interpret the results of the LLM
import inspect # needed to get the metadata on the functions used

MAX_ITERATIONS = 5
MODEL = "qwen3.5:2b"

# This is Abstraction Layer 2 - RAW PROMPT NO FUNCTION CALLING


# --- Tools ---


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
    price = float(price) # for RAW, the tool output is always a STRING, so it needs extra conversion step
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# no need for the JSON schema now, the tools now live inside the prompt as plain text
tools = {
    'get_product_price': get_product_price,
    'apply_discount': apply_discount,
}
# however, the LLM won't know how to use the functions, require a special function 
# to give it all the needed details like function itself, doctrings, signature
def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ is used to bypass the decorator wrappers
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions) # give to LLM as single text
# obtain tools description text
tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# react prompt - most important element, determines the ReAct loop architecture
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.
5. If the user does not specify a particular type or model of an item from the catalog, just pick a generic one, do NOT ask the user for specific name, type or model.
6. If you have all the info needed to reach final answer then write it following the format provided below.
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



# --- Helper: traced Ollama call ---
# without LangChain, LLM calls must be manually traced for LangSmith
# without independent tools, there is only the messages, where the tools now reside
@traceable(name = 'Raw Prompt Chat', run_type = 'llm')
def ollama_chat_traced(model, messages, options): # remove the tools since they are no longer independent from the messages
    return ollama.chat(model = model, messages = messages, options=options)


# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    ### REACT DIAGRAM ELEMENT: the Query
    prompt = react_prompt.format(question=question) # only a single prompt remains now
    scratchpad = "" # this will be used to store tool calling history
    ### REACT DIAGRAM ELEMENT: the Query

    print(f"Question: {question}")
    print("=" * 60)

    ### REACT DIAGRAM ELEMENT: the Agent Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # simulate memory by injecting the prompt + the scratchpad from the previous iteration
        full_prompt = prompt + scratchpad

        ### REACT DIAGRAM ELEMENT: the Thought
        # with Raw Prompt there will only be 1 message, containing the full prompt (prompt+scratchpad)
        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature":0} # this tells the LLM to stop producing text after it produces the "Obervation" token
        )
        # changed from ai_message to output to better highlight that now the ouput contains everything, not just the LLM's answer
        output = response.message.content  # this is an OLLAMA response object, different from Langchain's
        print(f"LLM Output:\n{output}")
        ### REACT DIAGRAM ELEMENT: the Thought

        ### REACT DIAGRAM ELEMENT: the Answer
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        ### REACT DIAGRAM ELEMENT: the Answer

        ### REACT DIAGRAM ELEMENT: the Tool
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")
        # with Raw there are no actual tool calls, so we need to parse through the output to see if any Action has been performed
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: could not parse Action/Action Input from LLM output"
            )
            break
        # if action/action input is found, then we can extract the tool details
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")
        # split comma-separated args; strip key= prefix if LLM output key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name} ({args})...")
        if tool_name not in tools: # make sure the LLM does not hallucinate a non-existent tool
            observation = f"ERROR: tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else: # run tool
            observation = str(tools[tool_name](*args)) # LLMs only digest STRING so the answer must be converted
        ### REACT DIAGRAM ELEMENT: the Tool

        ### REACT DIAGRAM ELEMENT: the Observation
        print(f"  [Tool Result] {observation}")
        # add tool use results to memory so that it can be reinjected into the LLM the next iteration
        scratchpad += f"{output}\nObservation: {observation}\nThought:" # NOTE: the memory needs to mirror the format of the original prompt
        ### REACT DIAGRAM ELEMENT: the Observation

    print("ERROR: Max iterations reached without a final answer")
    return None
    ### REACT DIAGRAM ELEMENT: the Agent Loop


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")