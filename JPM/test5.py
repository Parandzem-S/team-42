#!/usr/bin/env python3
"""
Earnings Call Transcript Processor using LLM Prompts
Extracts presentation and Q&A data from various transcript formats
"""

import pandas as pd
import json
import re, sys, os
from pathlib import Path
from openai import OpenAI

# Optional imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

def read_file(file_path: str) -> str:
    """Read any supported file format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        if not PyPDF2:
            raise ImportError("Install PyPDF2: pip install PyPDF2")
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return '\n'.join(page.extract_text() for page in reader.pages)
    
    elif ext in ['.docx', '.doc']:
        if not docx:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = docx.Document(file_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    
    else:  # Text file
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise Exception(f"Could not read {file_path}")

def extract_with_llm(text: str, api_key: str) -> tuple:
    """Extract using OpenAI API."""
    client = OpenAI(api_key=api_key)
    
    # Split into presentation and Q&A
    qa_start = text.find("QUESTION AND ANSWER SECTION")
    if qa_start == -1:
        qa_start = text.find("Q&A")
    
    if qa_start != -1:
        presentation = text[:qa_start]
        qa_section = text[qa_start:]
    else:
        presentation = text
        qa_section = ""
    
    # Clean presentation
    lines = [line for line in presentation.split('\n') 
             if not line.strip().startswith('Operator:')]
    clean_presentation = '\n'.join(lines)
    
    # Extract Q&A if exists
    qa_data = []
    if qa_section and len(qa_section) > 100:
        prompt = f"""
Extract Q&A data from this earnings call transcript and return ONLY valid JSON:

[
  {{
    "question_number": 1,
    "type": "question",
    "speaker_name": "John Doe",
    "speaker_details": "Analyst, ABC Securities", 
    "text": "My question is..."
  }},
  {{
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Jane CEO",
    "speaker_details": "Chief Executive Officer",
    "text": "Thank you for the question..."
  }}
]

Rules:
- Increment question_number for each new question
- Same question_number for corresponding answer
- Ignore Operator comments
- Return only JSON, no other text

Transcript:
{qa_section[:8000]}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if result.startswith("```json"):
                result = result.replace("```json", "").replace("```", "")
            
            try:
                qa_data = json.loads(result)
            except json.JSONDecodeError:
                print("Warning: Could not parse LLM JSON response")
                qa_data = []
                
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            qa_data = []
    
    return clean_presentation, qa_data

def simple_regex_extract(text: str) -> tuple:
    """Simple regex-based extraction fallback."""
    
    # Split into presentation and Q&A
    qa_start = text.find("QUESTION AND ANSWER SECTION")
    if qa_start == -1:
        qa_start = text.find("Q&A")
    
    if qa_start != -1:
        presentation = text[:qa_start]
        qa_section = text[qa_start:]
    else:
        presentation = text
        qa_section = ""
    
    # Clean presentation
    lines = [line for line in presentation.split('\n') 
             if not line.strip().startswith('Operator:')]
    clean_presentation = '\n'.join(lines)
    
    # Simple Q&A extraction
    qa_data = []
    if qa_section:
        lines = qa_section.split('\n')
        question_num = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for speaker pattern
            if (len(line.split()) <= 3 and 
                any(c.isupper() for c in line) and 
                i + 1 < len(lines)):
                
                speaker_name = line
                if i + 1 < len(lines):
                    title_line = lines[i + 1].strip()
                    
                    if title_line.endswith(' Q'):
                        question_num += 1
                        entry_type = 'question'
                        speaker_details = title_line[:-2]
                    elif title_line.endswith(' A'):
                        entry_type = 'answer'
                        speaker_details = title_line[:-2]
                    else:
                        i += 1
                        continue
                    
                    # Get text
                    text_lines = []
                    j = i + 2
                    while j < len(lines) and not (
                        len(lines[j].split()) <= 3 and 
                        j + 1 < len(lines) and
                        (lines[j + 1].endswith(' Q') or lines[j + 1].endswith(' A'))
                    ):
                        if lines[j].strip():
                            text_lines.append(lines[j].strip())
                        j += 1
                    
                    if text_lines:
                        qa_data.append({
                            'question_number': question_num,
                            'type': entry_type,
                            'speaker_name': speaker_name,
                            'speaker_details': speaker_details,
                            'text': ' '.join(text_lines)
                        })
                    
                    i = j
                else:
                    i += 1
            else:
                i += 1
    
    return clean_presentation, qa_data

def create_manual_prompts(text: str, output_dir: str):
    """Create manual prompts for use with any LLM."""
    
    qa_start = text.find("QUESTION AND ANSWER SECTION")
    if qa_start == -1:
        qa_start = text.find("Q&A")
    
    qa_section = text[qa_start:] if qa_start != -1 else "No Q&A section found"
    
    prompt = """
Extract Q&A data from this earnings call transcript and return as JSON:

[
  {
    "question_number": 1,
    "type": "question", 
    "speaker_name": "John Smith",
    "speaker_details": "Analyst, XYZ Securities",
    "text": "My question is about..."
  },
  {
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Jane Doe", 
    "speaker_details": "CEO",
    "text": "Thank you for the question..."
  }
]

Rules:
- Increment question_number for each new question
- Same question_number for the answer
- Ignore Operator comments
- Return only JSON

TRANSCRIPT:
[PASTE THE Q&A SECTION HERE]
"""
    
    # Save prompt and Q&A section
    with open(f"{output_dir}/manual_prompt.txt", 'w') as f:
        f.write(prompt)
    
    with open(f"{output_dir}/qa_section.txt", 'w') as f:
        f.write(qa_section)
    
    print(f"Manual prompt files created:")
    print(f"- {output_dir}/manual_prompt.txt")
    print(f"- {output_dir}/qa_section.txt")
    print("\nTo use: Copy prompt, replace placeholder with Q&A text, use in ChatGPT")

def process_transcript(file_path: str, api_key: str = None, output_dir: str = "."):
    """Main processing function."""
    
    print(f"Reading: {file_path}")
    text = read_file(file_path)
    print(f"Extracted {len(text)} characters")
    
    # Try LLM extraction if API key provided
    if api_key:
        try:
            print("Trying LLM extraction...")
            presentation, qa_data = extract_with_llm(text, api_key)
        except Exception as e:
            print(f"LLM failed: {e}")
            print("Falling back to regex...")
            presentation, qa_data = simple_regex_extract(text)
    else:
        print("No API key - using regex extraction...")
        presentation, qa_data = simple_regex_extract(text)
    
    # Create manual prompts regardless
    create_manual_prompts(text, output_dir)
    
    # Save results
    with open(f"{output_dir}/presentation.txt", 'w') as f:
        f.write(presentation)
    
    df = pd.DataFrame(qa_data) if qa_data else pd.DataFrame(
        columns=['question_number', 'type', 'speaker_name', 'speaker_details', 'text']
    )
    
    df.to_csv(f"{output_dir}/qa_data.csv", index=False)
    df.to_excel(f"{output_dir}/qa_data.xlsx", index=False)
    
    print(f"\nFiles created:")
    print(f"- presentation.txt")
    print(f"- qa_data.csv")
    print(f"- qa_data.xlsx")
    print(f"- manual_prompt.txt")
    print(f"- qa_section.txt")
    
    if not df.empty:
        print(f"\nExtracted {len(df)} Q&A entries")
        print(f"Questions: {len(df[df['type'] == 'question'])}")
        print(f"Answers: {len(df[df['type'] == 'answer'])}")
        print("\nFirst few entries:")
        print(df.head())
    else:
        print("\nNo Q&A data extracted - use manual prompts")
    
    return presentation, df

if __name__ == "__main__":
    # 1. Try LLM processing first
    try:
        file_name = "JPM_1q25-earnings-transcript.pdf"
        file_dir = "JPM/JPM Presentation texts/2025/Q1"
        file_path = os.path.join(file_dir, file_name)
        # check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            sys.exit(1) 
        API_KEY = os.environ.get("OPENAI_API_KEY")
        print(f"Using API key: {API_KEY[:5]}...{API_KEY[-5:]}")
        presentation, qa_df = process_transcript(file_path, API_KEY, "output")
    except:
        sys.exit("LLM processing failed. Please provide an OpenAI API key to try again.")