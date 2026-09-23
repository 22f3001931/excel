from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import csv
from pathlib import Path

app = FastAPI()

# Allow GET requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# CSV file should be in the same folder as this Python file
CSV_FILE = Path(__file__).parent / "students.csv"


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


@app.get("/api")
def get_students(class_: list[str] | None = Query(default=None, alias="class")):
    students = load_students()

    # No class query parameter: return all students
    if not class_:
        return {"students": students}

    # Use a set for efficient filtering while preserving CSV order
    requested_classes = set(class_)

    filtered_students = [
        student
        for student in students
        if student["class"] in requested_classes
    ]

    return {"students": filtered_students}


# Run locally with:
# uvicorn app:app --reload
