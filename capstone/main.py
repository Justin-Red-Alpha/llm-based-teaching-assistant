import exerciseformatter
import studentsolutionformatter

soln_file_path = "capture_me_solution.txt"
student_file_path = "submission.txt"

def main():
    ef = exerciseformatter.question_answer_extractor(soln_file_path)
    print(f"Extracted Question-Answer Pairs: {ef}")
    sf = studentsolutionformatter.parse_student_submission(student_file_path)
    print(f"Parsed Student Submission: {sf}")

if __name__ == "__main__":
    main()