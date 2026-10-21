import os
import csv
import requests

def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load environment variables FIRST so LangSmith can read the API key
load_env()

from langsmith import traceable

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

def get_c02_claim():
    csv_path = "data/aws_reinvent_candidate_items.csv"
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("candidate_id") == "C-02":
                return row.get("claim_text")
    raise ValueError("Candidate C-02 not found in CSV")

# The @traceable decorator captures the inputs and outputs and sends them to LangSmith
@traceable(run_type="llm", name="OpenRouter_Fact_Check")
def ask_openrouter(claim_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/reinvent-fact-checker",
        "X-Title": "Reinvent Fact Checker"
    }
    prompt = f"Is this a real AWS launch? Answer yes or no, and name the page that proves it.\n\nClaim: {claim_text}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    claim = get_c02_claim()
    print(f"Claim C-02: {claim}")
    print("\nAsking OpenRouter...")
    try:
        answer = ask_openrouter(claim)
        print("\nAnswer:")
        print(answer)
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
