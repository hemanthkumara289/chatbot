from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatGroq(model = "openai/gpt-oss-120b")

print("Welcome to the Chatbot! Type 'exit' to quit.")

messages = [
    SystemMessage(content="You are a funny AI assistant."),
]

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower() == "exit":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot: " + response.content)

print("Conversation history:")
for message in messages:
    if isinstance(message, SystemMessage):
        print("System: " + message.content)
    elif isinstance(message, HumanMessage):
        print("You: " + message.content)
    elif isinstance(message, AIMessage):
        print("Bot: " + message.content)