import os
import json
import boto3
from typing import Optional

# -------------------------
# Config
# -------------------------
USE_BEDROCK = True
MODEL_PROVIDER = "nova"

# Amazon Nova Micro (chat/text model)
NOVA_MODEL_ID = "arn:aws:bedrock:ap-south-1:417311687123:inference-profile/apac.amazon.nova-micro-v1:0"


# -------------------------
# Public function used by main.py
# -------------------------
def generate_answer(prompt: str, context: str) -> str:
    """
    Returns an answer from Bedrock (Nova Micro), optionally grounded by context.
    """
    if not USE_BEDROCK:
        return "Bedrock disabled by config"

    if MODEL_PROVIDER == "nova":
        return nova_bedrock(prompt, context)

    return "Local model disabled"


# -------------------------
# Bedrock - Nova Micro
# -------------------------
def nova_bedrock(prompt: str, context: Optional[str] = "") -> str:
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"

    client = boto3.client("bedrock-runtime", region_name=region)

    if context and context.strip():
        system_prompt = (
            "You are Siddh Guide, a helpful assistant.\n\n"
            "RULES:\n"
            "- Answer ONLY using the Context below.\n"
            "- If the answer is not clearly in the Context, reply exactly: I don't know.\n\n"
            f"Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are Siddh Guide, a helpful assistant.\n"
            "Answer the user clearly and briefly."
        )

    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "maxTokens": 400,
            "temperature": 0.2,
            "topP": 0.9
        }
    }

    response = client.invoke_model(
        modelId=NOVA_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())

    try:
        return result["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        pass

    try:
        return result["message"]["content"][0]["text"].strip()
    except Exception:
        pass

    return json.dumps(result)[:2000]
