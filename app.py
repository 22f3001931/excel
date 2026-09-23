from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
from pathlib import Path
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CSV_FILE = Path(__file__).parent / "students.csv"


# -------------------------
# Student API
# -------------------------

def load_students():
    students = []

    with open(CSV_FILE, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            students.append({
                "studentId": int(row["studentId"]),
                "class": row["class"]
            })

    return students


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/api")
def get_students(
    class_: list[str] | None = Query(default=None, alias="class")
):
    students = load_students()

    if not class_:
        return {"students": students}

    requested_classes = set(class_)

    filtered_students = [
        student
        for student in students
        if student["class"] in requested_classes
    ]

    return {"students": filtered_students}


# -------------------------
# Sentiment API
# -------------------------

class SentimentRequest(BaseModel):
    sentences: list[str]


positive_words = {
    "love", "loved", "lovely",
    "like", "liked", "likes",
    "great", "good", "excellent",
    "amazing", "awesome", "fantastic",
    "wonderful", "perfect", "best",
    "happy", "happiness",
    "enjoy", "enjoyed", "enjoying",
    "fun", "beautiful",
    "brilliant", "helpful",
    "success", "successful",
    "win", "winner",
    "excited", "exciting",
    "pleased", "glad",
    "thank", "thanks",
    "positive", "nice",
    "impressive", "impressed",
    "recommend", "recommended"
}

negative_words = {
    "hate", "hated", "horrible",
    "terrible", "bad", "awful",
    "worst", "sad", "sadness",
    "angry", "anger", "annoyed",
    "annoying", "disappointed",
    "disappointing", "dislike",
    "disliked", "poor", "useless",
    "broken", "failure", "failed",
    "fail", "problem", "problems",
    "error", "errors", "wrong",
    "pain", "painful", "boring",
    "boring", "hate", "horrible",
    "difficult", "frustrating",
    "frustrated", "negative",
    "complaint", "complaints",
    "sad", "cry", "crying"
}

# Words that reverse the meaning of the following word/phrase
negation_words = {
    "not", "never", "no", "isn't", "wasn't",
    "don't", "doesn't", "didn't", "can't",
    "couldn't", "won't", "wouldn't"
}


def classify_sentiment(sentence: str) -> str:
    text = sentence.lower()

    # Normalize punctuation
    words = re.findall(r"\b[\w']+\b", text)

    positive_score = 0
    negative_score = 0

    for i, word in enumerate(words):

        # Check whether the word is preceded by a negation
        negated = i > 0 and words[i - 1] in negation_words

        if word in positive_words:
            if negated:
                negative_score += 1
            else:
                positive_score += 1

        elif word in negative_words:
            if negated:
                positive_score += 1
            else:
                negative_score += 1

    if positive_score > negative_score:
        return "happy"

    if negative_score > positive_score:
        return "sad"

    return "neutral"


@app.post("/sentiment")
def sentiment(request: SentimentRequest):

    results = []

    for sentence in request.sentences:
        results.append({
            "sentence": sentence,
            "sentiment": classify_sentiment(sentence)
        })

    return {
        "results": results
    }
