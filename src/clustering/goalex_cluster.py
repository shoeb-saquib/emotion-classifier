import requests
import json

url = "https://genai.rcac.purdue.edu/api/chat/completions"
headers = {
    "Authorization": f"Bearer sk-8f827e402e6e424ba958363eb49bbc6c",
    "Content-Type": "application/json"
}
body = {
    "model": "llama3.1:latest",
    "messages": [
    {
        "role": "user",
        "content": "tell me about yourself"
    }
    ],
    "stream": False
}
response = requests.post(url, headers=headers, json=body)
if response.status_code == 200:
    data = json.loads(response.text)
    print(data['choices'][0]['message']['content'])
else:
    raise Exception(f"Error: {response.status_code}, {response.text}")