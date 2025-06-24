# This file contains YOUR EXACT PDF processing code from paste.txt

import re
import os
import pandas as pd
import pdfplumber
from typing import List, Dict, Any

# Internal titles pattern (dynamic, fully general) - YOUR CODE
internal_title_pattern = re.compile(
    r"(Group|Chief|Officer|CEO|CFO|Treasurer|Head|Director|Vice President|Finance|Investor Relations|Chairman|Chair|IR)",
    re.IGNORECASE
)

# Fully improved internal speaker detection - YOUR CODE
def is_internal_speaker(institution):
    if not institution or institution.strip() == "":
        return True
    if re.search(r"hsbc", institution, re.IGNORECASE):
        return True
    return bool(internal_title_pattern.search(institution))

def is_external_speaker(institution):
    return not is_internal_speaker(institution)

def extract_qa_from_pdf(pdf_path: str, file_label: str = None) -> List[Dict[str, Any]]:
    """
    YOUR EXACT extract_qa_from_pdf FUNCTION FROM paste.txt
    """
    qa_data = []
    current_speaker = ""
    current_institution = ""
    current_text = []

    speaker_institutions = {}
    question_number = -1
    current_question_owner = None
    in_qa_section = False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = pdf.pages[0].extract_text()

            # YEAR DETECTION
            year = None
            year_match = re.search(r"\b(20[1-3][0-9])\b", first_page_text)
            if year_match:
                year = int(year_match.group(1))

            # QUARTER DETECTION (fully robust)
            quarter = "Unknown"
            quarter_text = first_page_text.lower()

            post_results_match = re.search(r"post-results\s+([A-Za-z\s\-]+)", first_page_text, re.IGNORECASE)
            if post_results_match:
                quarter = "Post-Results " + post_results_match.group(1).strip()
            elif re.search(r"\bq\s*4\b", quarter_text):
                quarter = 4
            elif re.search(r"\bq\s*3\b", quarter_text):
                quarter = 3
            elif re.search(r"\bq\s*2\b", quarter_text):
                quarter = 2
            elif re.search(r"\bq\s*1\b", quarter_text):
                quarter = 1
            elif re.search(r"interim", quarter_text) and re.search(r"\bh[\s\-]*1\b", quarter_text):
                quarter = "Interim/H1"
            elif re.search(r"interim", quarter_text):
                quarter = "Interim"
            elif re.search(r"\bh[\s\-]*1\b", quarter_text):
                quarter = "H1"
            elif re.search(r"\bannual\b|\bfull\s*year\b", quarter_text):
                quarter = "Annual"

            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split("\n")

                for line in lines:
                    # Q&A section detection
                    line_clean = re.sub(r"[\W_]+", "", line.lower())
                    if not in_qa_section and re.search(r"(qa|questionsandanswers|questions)", line_clean):
                        in_qa_section = True
                        continue

                    # Flexible speaker matching
                    match_full = re.match(r"^([A-Z\s\.]+),\s*([A-Za-z\s&]+):\s*(.*)", line)
                    if match_full:
                        speaker = match_full.group(1).title()
                        institution = match_full.group(2).title()
                        remainder = match_full.group(3).strip()
                    else:
                        match_short = re.match(r"^([A-Z\s\.]+):\s*(.*)", line)
                        if match_short:
                            speaker = match_short.group(1).title()
                            institution = speaker_institutions.get(speaker, "")  # Use previous institution if known
                            remainder = match_short.group(2).strip()
                        else:
                            if line.strip():
                                current_text.append(line.strip())
                            continue

                    # Save previous block
                    if current_text:
                        flag_question = (
                            in_qa_section and current_speaker == current_question_owner
                        )
                        presentation = 1 if question_number in [None, -1] else 0

                        qa_data.append({
                            "File": file_label or os.path.basename(pdf_path),
                            "Bank Name": "HSBC",
                            "Year": year,
                            "Quarter": quarter,
                            "Speaker name": current_speaker,
                            "Institution": current_institution,
                            "Speaker text": " ".join(current_text),
                            "flag_question": flag_question,
                            "Question No": question_number if in_qa_section else None,
                            "presentation": presentation
                        })
                        current_text = []

                    # Update current speaker
                    current_speaker = speaker
                    current_institution = institution
                    speaker_institutions[current_speaker] = current_institution

                    # Dynamically switch to Q&A when first external detected
                    if not in_qa_section and is_external_speaker(current_institution):
                        in_qa_section = True

                    # 🔥 FULLY FIXED QUESTION NUMBERING LOGIC 🔥
                    if in_qa_section:
                        if is_external_speaker(current_institution):
                            if current_question_owner != current_speaker:
                                question_number += 1
                                current_question_owner = current_speaker

                    if remainder:
                        current_text.append(remainder)

            # Save final block
            if current_text:
                flag_question = (
                    in_qa_section and current_speaker == current_question_owner
                )
                presentation = 1 if question_number in [None, -1] else 0

                qa_data.append({
                    "File": file_label or os.path.basename(pdf_path),
                    "Bank Name": "HSBC",
                    "Year": year,
                    "Quarter": quarter,
                    "Speaker name": current_speaker,
                    "Institution": current_institution,
                    "Speaker text": " ".join(current_text),
                    "flag_question": flag_question,
                    "Question No": question_number if in_qa_section else None,
                    "presentation": presentation
                })

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    return qa_data

def process_all_pdfs(root_dir: str) -> pd.DataFrame:
    """
    YOUR EXACT process_all_pdfs FUNCTION
    """
    all_results = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                print(f"Processing: {rel_path}")
                qa_rows = extract_qa_from_pdf(full_path, file_label=rel_path)
                all_results.extend(qa_rows)

    df_all = pd.DataFrame(all_results)
    df_all = df_all[df_all['Speaker name'].str.strip() != ""]
    return df_all