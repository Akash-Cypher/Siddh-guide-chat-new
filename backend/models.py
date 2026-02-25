import os
import json
import boto3
from typing import Optional

# -------------------------
# Config
# -------------------------
USE_BEDROCK = True
MODEL_PROVIDER = "nova"

# Move model ID to env (more flexible for prod)
NOVA_MODEL_ID = os.getenv(
    "NOVA_MODEL_ID",
    "arn:aws:bedrock:ap-south-1:417311687123:inference-profile/apac.amazon.nova-micro-v1:0"
)

# -------------------------
# Create Bedrock client ONCE (important for prod)
# -------------------------
_BEDROCK_CLIENT = None

def _get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-south-1"
        _BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name=region)
    return _BEDROCK_CLIENT


# -------------------------
# Public function used by main.py
# -------------------------
def generate_answer(prompt: str, context: str) -> str:
    if not USE_BEDROCK:
        return "Bedrock disabled by config"

    if MODEL_PROVIDER == "nova":
        return nova_bedrock(prompt, context)

    return "Local model disabled"


# -------------------------
# Bedrock - Nova Micro
# -------------------------
def nova_bedrock(prompt: str, context: Optional[str] = "") -> str:
    client = _get_bedrock_client()

    if context and context.strip():
        system_prompt = (
            "You are Siddh Guide, a warm and respectful assistant for Siddhanta Knowledge Foundation.\n\n"
            "STRICT RULES (must follow):\n"
            "- NEVER output the phrases: \"I don't know\", \"I dont know\", \"I do not know\".\n"
            "- Use ONLY the Context below for factual claims.\n"
            "- If the user request is broad/vague:\n"
            "  - Ask EXACTLY ONE short follow-up question.\n"
            "- If the user request is specific AND Context contains matches:\n"
            "  - Suggest up to 3 relevant courses with short reasons.\n"
            "- If the answer is not in the Context:\n"
            "  - Politely say you don’t have that info in the current course database.\n"
            "  - Ask EXACTLY ONE short follow-up question.\n"
            "- Keep responses short (2–4 sentences).\n\n"
            f"Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are Siddh Guide, a warm and respectful assistant.\n\n"
            "STRICT RULES:\n"
            "- NEVER output: \"I don't know\".\n"
            "- If vague, ask ONE short follow-up question.\n"
            "- If missing info, politely say it’s not in the current course list.\n"
            "- Keep responses short (1–3 sentences).\n"
        )

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "maxTokens": 120,
            "temperature": 0.2,
            "topP": 0.9
        },
    }

    response = client.invoke_model(
        modelId=NOVA_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
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