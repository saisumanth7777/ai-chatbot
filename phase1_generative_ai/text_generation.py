"""
Phase 1 — Text Generation with Claude API
Run: python phase1_generative_ai/text_generation.py
Needs: ANTHROPIC_API_KEY in .env
"""
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def basic_generation(user_query: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": user_query}]
    )
    return response.content[0].text


def streaming_generation(user_query: str) -> None:
    print("Claude (streaming): ", end="", flush=True)
    with client.messages.stream(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": user_query}]
    ) as stream:
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
    print()


def engineered_prompt(user_query: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=(
            "You are a helpful AI tutor teaching a beginner about AI concepts. "
            "Always: (1) explain in simple language, (2) give a real-world analogy, "
            "(3) end with one follow-up question to deepen understanding."
        ),
        messages=[{"role": "user", "content": user_query}]
    )
    return response.content[0].text


def multi_turn_chat() -> None:
    history = []
    turns = [
        "What is a neural network?",
        "Can you give me a Python code example of the simplest possible one?",
    ]
    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        history.append({"role": "user", "content": user_msg})
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=history,
        )
        assistant_msg = response.content[0].text
        print(f"Claude: {assistant_msg}")
        history.append({"role": "assistant", "content": assistant_msg})


if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Basic generation")
    print("=" * 60)
    reply = basic_generation("What is generative AI in one sentence?")
    print(f"Claude: {reply}\n")

    print("=" * 60)
    print("EXAMPLE 2: Streaming")
    print("=" * 60)
    streaming_generation("Explain what a GAN is in 2 sentences.")
    print()

    print("=" * 60)
    print("EXAMPLE 3: Engineered prompt")
    print("=" * 60)
    reply = engineered_prompt("What is backpropagation?")
    print(f"Claude: {reply}\n")

    print("=" * 60)
    print("EXAMPLE 4: Multi-turn conversation")
    print("=" * 60)
    multi_turn_chat()
