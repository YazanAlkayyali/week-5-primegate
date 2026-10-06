from langgraph_agent import ask

print("ask away (type exit to leave)")

while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    response = ask(user_input)
    print(f"Bot answer: {response}")