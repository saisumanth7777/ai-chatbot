"""
Phase 3 — Tool Use (Function Calling)
Run: python phase3_model_dev/tool_use.py
Needs: ANTHROPIC_API_KEY in .env
"""
import datetime
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %B %d %Y at %I:%M %p")


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    fake_weather = {
        "new york": "Sunny, 72°F (22°C), humidity 45%",
        "london":   "Cloudy, 58°F (14°C), light rain expected",
        "tokyo":    "Partly cloudy, 68°F (20°C), humidity 60%",
        "sydney":   "Clear skies, 77°F (25°C), humidity 50%",
        "paris":    "Overcast, 61°F (16°C), chance of showers",
    }
    return fake_weather.get(city.lower(), f"Weather data not available for '{city}'.")


TOOLS = [
    {
        "name": "get_current_time",
        "description": "Returns the current date and time. Use this when the user asks what time or date it is.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "calculate",
        "description": "Evaluates a math expression and returns the result. Use this for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '2 + 2' or '100 * 0.15'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Returns the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'London' or 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    },
]


def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "calculate":
        return calculate(tool_input["expression"])
    elif tool_name == "get_weather":
        return get_weather(tool_input["city"])
    return f"Unknown tool: {tool_name}"


def demonstrate_tool_flow(user_question: str) -> str:
    print(f"\nUser: {user_question}")
    print("-" * 40)

    messages = [{"role": "user", "content": user_question}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        tools=TOOLS,
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        tool_name  = tool_use_block.name
        tool_input = tool_use_block.input

        print(f"[Claude wants to call: {tool_name}({tool_input})]")
        tool_result = run_tool(tool_name, tool_input)
        print(f"[Tool returned: {tool_result}]")

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": tool_result}]
        })

        final_response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            tools=TOOLS,
            messages=messages,
        )
        reply = final_response.content[0].text
    else:
        reply = response.content[0].text

    print(f"Claude: {reply}")
    return reply


def tool_chatbot() -> None:
    print("\nTool-powered chatbot! Claude can check time, do math, and get weather.")
    print("Type 'quit' to exit.")
    print("-" * 40)

    history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            tools=TOOLS,
            messages=history,
        )

        while response.stop_reason == "tool_use":
            tool_use_block = next(b for b in response.content if b.type == "tool_use")
            print(f"  [Using tool: {tool_use_block.name}]")
            tool_result = run_tool(tool_use_block.name, tool_use_block.input)

            history.append({"role": "assistant", "content": response.content})
            history.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": tool_result}]
            })
            response = client.messages.create(
                model=MODEL,
                max_tokens=500,
                tools=TOOLS,
                messages=history,
            )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"\nClaude: {reply}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Tool Use (Function Calling)")
    print("=" * 60)
    print("""
Choose an example:
  1 — See the tool flow step-by-step (3 demo questions)
  2 — Full tool-powered chatbot
""")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        demonstrate_tool_flow("What is today's date and time?")
        demonstrate_tool_flow("What is 15% of 847?")
        demonstrate_tool_flow("What's the weather like in Tokyo right now?")
    elif choice == "2":
        tool_chatbot()
    else:
        print("Invalid choice. Running chatbot by default.")
        tool_chatbot()
