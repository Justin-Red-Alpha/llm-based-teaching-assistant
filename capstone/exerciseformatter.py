import os
import io
from PyPDF2 import PdfReader
from docx import Document
import re

def upload_file(file_path):
    if file_path:
        try:
            with open(file_path, 'rb') as f:
                file_content_bytes = f.read()
            uploaded = {file_path: file_content_bytes}
            print(f"Using file '{file_path}' from local path.")
        except FileNotFoundError:
            file_path = None
            uploaded = {}
            print(f"File '{file_path}' not found. Please check the path.")
        except Exception as e:
            file_path = None
            uploaded = {}
            print(f"An error occurred while reading the file: {e}")
    else:
        uploaded = {}
        print("No file path provided.")

    if file_path and uploaded:
        print(f"File '{file_path}' is ready for processing.")
    else:
        print("No file available for processing.")
    return uploaded


def format_exercise(uploaded_file_name):
    extracted_text = ""
    file_type = None 
    uploaded = upload_file(uploaded_file_name)
    if uploaded_file_name:
        file_extension = os.path.splitext(uploaded_file_name)[1].lower()

        if file_extension == '.txt':
            file_type = "text file"
            extracted_text = uploaded[uploaded_file_name].decode('utf-8')
            print(f"Identified file type: {file_type}")

        elif file_extension == '.pdf':
            file_type = "PDF document"
            try:
                pdf_file = io.BytesIO(uploaded[uploaded_file_name])
                reader = PdfReader(pdf_file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    extracted_text += page.extract_text() + "\n"
                print(f"Identified file type: {file_type}")
            except Exception as e:
                print(f"Error processing PDF file: {e}")
                extracted_text = None

        elif file_extension == '.docx':
            file_type = "DOCX document"
            try:
                doc_file = io.BytesIO(uploaded[uploaded_file_name])
                document = Document(doc_file)
                for para in document.paragraphs:
                    extracted_text += para.text + "\n"
                print(f"Identified file type: {file_type}")
            except Exception as e:
                print(f"Error processing DOCX file: {e}")
                extracted_text = None

        else:
            file_type = "unsupported type"
            print(f"Unsupported file type: {file_extension}")
            extracted_text = None

    else:
        print("No file was uploaded to process.")
        extracted_text = None

    if extracted_text:
        print("\n--- Extracted Text (first 500 characters) ---")
        print(extracted_text[:500])
        print("---------------------------------------------")
    else:
        print("\nNo text extracted or file type unsupported.")
    return extracted_text

def question_answer_extractor(uploaded_file_name):
    extracted_text = format_exercise(uploaded_file_name)
    questions_answers = []
    # Split the content into blocks based on double newlines (CRLF), then clean up
    blocks = [block.strip() for block in extracted_text.strip().split('\r\n\r\n') if block.strip()]

    for block in blocks:
        # Find the first newline to separate question from answer
        first_newline_index = block.find('\r\n')
        if first_newline_index != -1:
            raw_question = block[:first_newline_index].strip()
            answer = block[first_newline_index+2:].strip() # +2 to account for '\r\n'

            # Extract question number and question text
            match = re.match(r'(\d+)\)\s*(.*)', raw_question)
            if match:
                question_number = match.group(1)
                question_text = match.group(2).strip()
                questions_answers.append({"question_id": question_number, "question": question_text, "answer": answer})
            else:
                # Handle cases where question format might be unexpected
                questions_answers.append({"question_id": None, "question": raw_question, "answer": answer})
        else:
            # Handle blocks that might just be a question with no apparent answer on a new line
            match = re.match(r'(\d+)\)\s*(.*)', block)
            if match:
                questions_answers.append({"question_id": match.group(1), "question": match.group(2).strip(), "answer": None})
            else:
                questions_answers.append({"question_id": None, "question": block, "answer": None})

    print("Extracted Question-Answer Pairs:")
    print("---------------------------------")
    for qa in questions_answers:
        print(f"Question Number: {qa['question_id']}")
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer']}\n")

    print(f"Successfully extracted {len(questions_answers)} question-answer pairs.")
    return questions_answers