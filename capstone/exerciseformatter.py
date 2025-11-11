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
    lines = extracted_text.splitlines()

    questions_answers = []
    current_combined_q_text_parts = []
    current_combined_a_lines = []
    active_main_q_id = None  # Reset active main question ID
    sub_question_alpha_counter = 0

    # --- Nested helper function ---
    def finalize_current_main_qa():
        """Saves the current combined question and its accumulated answer to questions_answers."""
        if current_combined_q_text_parts:  # Only save if there's a question being built
            questions_answers.append({
                "question_id": active_main_q_id,
                "question": "\n".join(current_combined_q_text_parts).strip(),
                "answer": "\n".join(filter(None, current_combined_a_lines)).strip()
            })
            # Reset accumulators
            current_combined_q_text_parts.clear()
            current_combined_a_lines.clear()

    # Pre-compile regex patterns once (outside the loop)
    numbered_question_pattern = re.compile(r'^\s*(\d+)\)\s*(.*)')
    question_like_pattern = re.compile(
        r'(?:What|Who|When|Where|Why|How|Which|Can|Do|Is|Are|Will)\b.*[?]?$', 
        re.IGNORECASE
    )

    # --- Main parsing loop ---
    for line in lines:
        line = line.strip()

        if not line:  # Preserve blank lines within answers
            if active_main_q_id is not None or current_combined_q_text_parts:
                current_combined_a_lines.append("")
            continue

        numbered_match = numbered_question_pattern.match(line)
        question_like_match = question_like_pattern.match(line)

        if numbered_match:
            # New main question starts → finalize previous one
            finalize_current_main_qa()

            active_main_q_id = numbered_match.group(1)
            sub_question_alpha_counter = 0  # Reset for new main question
            current_combined_q_text_parts.append(numbered_match.group(2))

        elif question_like_match:
            if active_main_q_id is None:
                # Standalone question
                finalize_current_main_qa()
                current_combined_q_text_parts.append(line)
            else:
                # Sub-question within main question
                sub_question_alpha_counter += 1
                sub_id_suffix = chr(ord('a') + sub_question_alpha_counter - 1)
                sub_question_label = f"{active_main_q_id}{sub_id_suffix})"
                current_combined_q_text_parts.append(f"{sub_question_label} {line}")

        elif active_main_q_id is not None or current_combined_q_text_parts:
            # Line is part of an answer
            current_combined_a_lines.append(line)
        else:
            # Irrelevant text (intro, etc.)
            pass

    # Finalize last Q&A
    finalize_current_main_qa()

    return questions_answers