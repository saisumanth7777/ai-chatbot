"""
Phase 2 — Conversation Memory
Run: python phase2_llm/conversation_memory.py
Needs: ANTHROPIC_API_KEY in .env
"""
import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"
HISTORY_FILE = "phase2_llm/chat_history.json"


def chat_loop() -> None:
    print("\nChatbot started! Type 'quit' to exit.")
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
            messages=history,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"\nClaude: {reply}")


def trim_history(history: list, max_turns: int) -> list:
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history


def chat_loop_with_limit(max_turns: int = 5) -> None:
    print(f"\nChatbot with sliding window (remembers last {max_turns} turns).")
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
        trimmed = trim_history(history, max_turns)
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=trimmed,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"\nClaude: {reply}")


def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[History saved to {HISTORY_FILE}]")


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def persistent_chat(max_turns: int = 10) -> None:
    history = load_history()

    if history:
        print(f"\nWelcome back! Resuming conversation ({len(history) // 2} previous turns).")
    else:
        print("\nNew conversation started.")
    print("Commands: 'quit' to save & exit | 'clear' to wipe memory")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            save_history(history)
            print("Conversation saved. See you next time!")
            break

        if user_input.lower() == "clear":
            history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            print("[Memory cleared. Starting fresh.]")
            continue

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        trimmed = trim_history(history, max_turns)
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=trimmed,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"\nClaude: {reply}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: Conversation Memory")
    print("=" * 60)
    print("""
Choose an example to run:
  1 — Basic chat loop (unlimited memory)
  2 — Chat with sliding window (remembers last 5 turns)
  3 — Persistent chat (saves to disk, resumes next run)
""")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        chat_loop()
    elif choice == "2":
        chat_loop_with_limit(max_turns=5)
    elif choice == "3":
        persistent_chat(max_turns=10)
    else:
        print("Invalid choice. Running persistent chat by default.")
        persistent_chat()
