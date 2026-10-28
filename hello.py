import os
from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["LLM_API_KEY"],
)

# List the models your key can use
for m in client.models.list():
    print(m.id)

# Send one message
response = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        {"role": "system", "content": "You are an analog circuit design tutor."},
        {"role": "user", "content": "Explain the -3 dB cutoff of an RC filter in 2 sentences."},
    ],
)
print("\n" + response.choices[0].message.content)