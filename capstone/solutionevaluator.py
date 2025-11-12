import json
import os
import re
import csv
import studentsolutionformatter
from openai_client import client

# add key here as variable name 'client'
# client =

def extract_user_answer(user_input, question_text_to_remove=None) -> str:
    # Ensure user_input is a string before attempting .strip()
    text = str(user_input).strip()

    if question_text_to_remove:
        # Remove repeated question if it's at the beginning of the user's answer
        # Use re.escape to handle special characters in the question text
        question_in_answer = re.escape(question_text_to_remove)
        # Remove only once if it's at the very beginning (case-insensitive)
        text = re.sub(rf'(?is)^{question_in_answer}\s*\??', '', text, 1).strip()

    text = re.sub(r'(?i)(filter:.*|status bar.*)', '', text).strip()     # remove UI text
    text = re.sub(r'(?i)(ans\s*[:\-]?\s*)', '', text).strip()            # remove "Ans:"
    return text

# --- LLM evaluator ---
def llm_eval(question, user_answer, reference_answer):
    prompt = f"""
You are grading a network analysis question.

Question: {question}
Correct Answer: {reference_answer}
User Answer: {user_answer}

Evaluate the user's answer in JSON with:
- correctness: ["correct", "partially correct", "incorrect"]
- score: 0 to 1
- explanation: short reason
"""
    response = client.chat.completions.create(
        model="gpt-4o", # Using a more capable model for evaluation
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Add checks for empty response or choices
    if not response or not response.choices:
        print("Warning: LLM response or choices list is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned an empty response or no choices."
        }

    llm_content = response.choices[0].message.content
    if not llm_content:
        print("Warning: LLM message content is empty.")
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": "LLM returned empty message content."
        }

    # Extract JSON from markdown code block if present
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = llm_content # Assume it's pure JSON if no markdown block

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Raw LLM response (attempted parse): {json_string}")
        print(f"Original LLM content: {llm_content}")
        # Return a default 'incorrect' evaluation to allow processing to continue
        return {
            "correctness": "incorrect",
            "score": 0.0,
            "explanation": f"LLM returned malformed JSON. Original response: {llm_content[:100]}..."
        }

# --- Evaluate student submissions ---
def evaluate_student_submission(merged_data):
    results = []

    for question_id, question_text, model_ans, student_ans_raw in merged_data:
        # question_id, question_text, model_ans, student_ans_raw are now directly available from merged_data

        # Pass the specific question_text to extract_user_answer for removal
        clean_answer = extract_user_answer(student_ans_raw, question_text)

        # Always use LLM evaluation
        if len(clean_answer) >= 1:
            result = llm_eval(question_text, clean_answer, model_ans)
        else:
            result = llm_eval(question_text, student_ans_raw, model_ans)

        results.append({
            "question_id": question_id,
            "question": question_text,
            "model_answer": model_ans,
            "raw_user_answer": student_ans_raw,
            "clean_user_answer": clean_answer,
            **result
        })
    return results

def Write_Student_Evaluation(student_evaluation_list: list, student_name: str, base_output_dir: str = "submissions"):
  # Construct the directory path for the student
  student_output_dir = os.path.join(base_output_dir, student_name)

  # Ensure the student's directory exists
  os.makedirs(student_output_dir, exist_ok=True)

  # Construct the full filepath for the evaluation CSV
  filepath = os.path.join(student_output_dir, "evaluation.csv")

  with open(filepath,"w", newline = "", encoding="utf-8") as file:
      writer = csv.DictWriter(file, fieldnames=student_evaluation_list[0].keys())
      writer.writeheader()
      writer.writerows(student_evaluation_list)

def batch_process_student_submissions(student_submissions_data : list[str], questions_answers : list[dict]):
    for student in student_submissions_data:
        merged = []
        student_answer_eval = []
        student_data = studentsolutionformatter.parse_student_submission(student['submission_path'])
        for q in questions_answers:
            q_num = str(q['question_id']) + '.'  # match student_submission keys
            question_text = q['question']
            model_ans = q['answer']
            student_ans = student_data['answers'].get(q_num, "")  # fallback empty string if missing
            merged.append((q_num, question_text, model_ans, student_ans)) # Include question_id

        # --- Run evaluation ---
        # Now passing only merged directly to the evaluation function
        results = evaluate_student_submission(merged)

        # print("-----", student['student_name'], "-----")
        for r in results:
            if (r['question_id'] == "None."):
                continue

            student_answer_eval.append({
                # "student_name": student['student_name'],
                "question_id": r['question_id'],
                "answer": r['raw_user_answer'],
                "correctness": r['correctness'],
                "score": r['score'],
                "explanation": r['explanation']
            })
        Write_Student_Evaluation(student_answer_eval, student['student_name'])


