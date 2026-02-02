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
    "You are Siddh Guide, a warm and respectful assistant for Siddhanta Knowledge Foundation.\n\n"
    "RULES:\n"
    "- Use ONLY the Context below for factual claims.\n"
    "- If the answer is not in the Context, do NOT say 'I don't know.' as a standalone reply.\n"
    "- Instead: say you don’t have that info yet in a polite tone, then offer help.\n"
    "- Offer ONE of these:\n"
    "  (a) ask 1 short follow-up question, OR\n"
    "  (b) suggest up to 3 relevant courses/topics from the Context.\n"
    "- Keep responses short and natural (2–5 sentences).\n"
    "- If suggesting courses, list max 3 with a short reason (5–10 words).\n\n"
    f"Context:\n{context}"
)
    else:
        system_prompt = (
            "You are Siddh Guide, a helpful assistant.\n"
            "Answer the user clearly."
            "Keep answers short: 1-2 sentences. No filler."
        )

    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "maxTokens": 100,
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
