from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Cek isi variabel env
print("KEY:", os.getenv("OPENROUTER_API_KEY"))
print("URL:", os.getenv("OPENROUTER_BASE_URL"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENROUTER_BASE_URL")

llm = ChatOpenAI(model="mistralai/mistral-7b-instruct", temperature=0.3)
print(llm.invoke("Siapa presiden Indonesia?").content)
 