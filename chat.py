import asyncio
from langchain_agent import ask


async def main():
    print("ask away (type exit to leave)")
    while True:
        user_input = input("you: ")
        if user_input.lower() == "exit":
            break
        response = await ask(user_input)
        print(f"Bot answer: {response}")


if __name__ == "__main__":
    asyncio.run(main())