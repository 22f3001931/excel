import os
import json
import requests

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_MODEL = "openai/gpt-4.1-nano"
VALID = {"happy", "sad", "neutral"}


def classify_batch_with_ai(sentences: list[str]) -> list[str]:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    resp = requests.post(
        AIPIPE_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={
            "model": AIPIPE_MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify the emotional sentiment of each sentence as exactly one of: "
                        "happy (positive, joyful, excited, grateful, pleased), "
                        "sad (negative: sadness, anger, frustration, disappointment, complaints), "
                        "neutral (factual, no clear emotion). "
                        "Return one label per sentence, in the same order."
                    ),
                },
                {"role": "user", "content": numbered},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiments",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiments": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["happy", "sad", "neutral"]},
                            }
                        },
                        "required": ["sentiments"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    labels = [str(x).lower().strip() for x in data["sentiments"]]
    if len(labels) != len(sentences) or any(l not in VALID for l in labels):
        raise ValueError("Bad AI output")
    return labels


@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    sentences = request.sentences
    try:
        labels = classify_batch_with_ai(sentences)
    except Exception as e:
        print("AI sentiment failed, using rules:", repr(e))
        labels = [classify_sentiment(s) for s in sentences]

    return {
        "results": [
            {"sentence": s, "sentiment": l} for s, l in zip(sentences, labels)
        ]
    }
