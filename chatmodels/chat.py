from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model = "openai/gpt-oss-120b",temperature=0.9,max_tokens=200)

response = model.invoke("who are you?")

print(response.content)