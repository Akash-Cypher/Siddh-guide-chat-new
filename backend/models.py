import os
import json
import boto3

USE_BEDROCK = True
MODEL_PROVIDER = "titan"
TITAN_MODEL_ID = "amazon.titan-text-express-v1"

def generate_answer(prompt: str, context: str) -> str:
    if not USE_BEDROCK:
        return "Bedrock disabled by config"

    if MODEL_PROVIDER == "titan":
        return titan_bedrock(prompt, context)
    else:
        return "Local model disabled"

def titan_bedrock(prompt: str, context: str) -> str:
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )

    if context and context.strip():
        full_prompt = f"""
You are Sidh Guide, a helpful assistant.

RULES:
- Answer ONLY using the Context below.
- If the answer is not clearly in the Context, reply exactly: I don't know.

Context:
{context}

Question:
{prompt}

Answer:
""".strip()
    else:
        full_prompt = f"""
You are Sidh Guide, a helpful assistant.
Answer the user clearly and briefly.

User question:
{prompt}
""".strip()

    body = {
        "inputText": full_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.2,
            "topP": 0.9
        }
    }

    response = client.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    return result["results"][0]["outputText"].strip()
