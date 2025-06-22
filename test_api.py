from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

try:
    chat = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        temperature=0.2,
        max_tokens=10,
        timeout=10
    )
    print(" Response:", chat.choices[0].message.content)
except Exception as e:
    print(" Failed:", e)
