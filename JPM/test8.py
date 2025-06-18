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

def fix_pdf_encoding(text: str) -> str:
    """Fix common PDF encoding issues."""
    
    # Common PDF encoding problems and their fixes
    encoding_fixes = {
        'â€"': '—',  # em dash
        'â€"': '—',  # em dash variant
        'â€™': "'",  # right single quotation mark
        'â€œ': '"',  # left double quotation mark
        'â€': '"',   # right double quotation mark
        'â€¢': '•',  # bullet point
        'â€¦': '…',  # horizontal ellipsis
        'â€˜': "'",  # left single quotation mark
        'Â': '',     # non-breaking space artifacts
        'â€‹': '',   # zero-width space
        'ï¿½': '',   # replacement character
        'â€Š': ' ',  # thin space
        'â€‰': ' ',  # thin space variant
        'â€ˆ': ' ',  # punctuation space
        'â€‡': ' ',  # figure space
    }
    
    # Apply fixes
    for bad_char, good_char in encoding_fixes.items():
        text = text.replace(bad_char, good_char)
    
    # Additional cleanup
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters that didn't get fixed
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text

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
            text = '\n'.join(page.extract_text() for page in reader.pages)
            
        # Fix common PDF encoding issues
        text = fix_pdf_encoding(text)
        return text
    
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

def clean_presentation_text(text: str) -> str:
    """Clean presentation text by removing operators, extra spaces, dots, and blank lines."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip operator comments
        if line.startswith('Operator:') or line.startswith('Operator '):
            continue
            
        # Skip lines that are mostly dots or dashes (separators)
        if re.match(r'^[.\-=_\s]{10,}$', line):
            continue
            
        # Skip empty lines
        if not line:
            continue
            
        # Remove excessive dots at the end of lines
        line = re.sub(r'\.{3,}.*$', '', line).strip()
        
        # Remove standalone dots or single characters
        if len(line) <= 2 and line in ['.', '..', '-', '--', '=', '==']:
            continue
            
        if line:
            cleaned_lines.append(line)
    
    # Join lines and remove excessive whitespace
    result = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive newlines (keep max 2 for paragraph breaks)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # Remove multiple spaces
    result = re.sub(r' {2,}', ' ', result)
    
    return result.strip()

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
    
    # Clean presentation thoroughly
    clean_presentation = clean_presentation_text(presentation)
    
    # Extract Q&A if exists
    qa_data = []
    if qa_section and len(qa_section) > 100:
        prompt = f"""
Extract ALL Q&A data from this earnings call transcript. Note that multiple people may answer the same question.

Return ONLY valid JSON array with ALL questions and answers:

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
  }},
  {{
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Bob CFO",
    "speaker_details": "Chief Financial Officer",
    "text": "To add to that..."
  }},
  {{
    "question_number": 2,
    "type": "question",
    "speaker_name": "Mary Smith",
    "speaker_details": "Analyst, XYZ Bank",
    "text": "My next question..."
  }}
]

Rules:
- Extract EVERY question and EVERY answer
- Multiple people can answer same question (same question_number)
- Increment question_number only for NEW questions
- Ignore ALL Operator comments completely
- Return only JSON, no other text

Transcript:
{qa_section[:10000]}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if result.startswith("```json"):
                result = result.replace("```json", "").replace("```", "")
            
            # Remove any text before [ or after ]
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                result = result[json_start:json_end]
            
            try:
                qa_data = json.loads(result)
                print(f"LLM extracted {len(qa_data)} Q&A entries")
            except json.JSONDecodeError:
                print("Warning: Could not parse LLM JSON response")
                qa_data = []
                
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            qa_data = []
    
    return clean_presentation, qa_data

def extract_qa_alternative_method(qa_section: str) -> list:
    """Alternative method to extract Q&A when standard method doesn't work well."""
    
    qa_data = []
    
    # Look for any pattern that might indicate speakers
    # Split by lines that look like names (short lines with capital letters)
    lines = qa_section.split('\n')
    
    question_num = 0
    current_question_num = 0
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and separator lines
        if not line or re.match(r'^[.\-=_\s]{3,}$', line):
            i += 1
            continue
            
        # Look for potential speaker names (short lines, mostly letters)
        if (len(line.split()) <= 4 and 
            len(line) < 50 and
            any(c.isalpha() for c in line) and
            not line.lower().startswith('operator')):
            
            speaker_name = line
            
            # Look ahead for title/role information
            speaker_details = ""
            entry_type = ""
            
            # Check next few lines for role/type information
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                    
                # Check if this line indicates question or answer
                if (any(word in next_line.lower() for word in ['analyst', 'research', 'bank', 'securities', 'investment']) or
                    next_line.endswith(' Q')):
                    question_num += 1
                    current_question_num = question_num
                    entry_type = 'question'
                    speaker_details = next_line.replace(' Q', '').strip()
                    break
                elif (any(word in next_line.lower() for word in ['ceo', 'cfo', 'chief', 'president', 'officer', 'director']) or
                      next_line.endswith(' A')):
                    entry_type = 'answer'
                    speaker_details = next_line.replace(' A', '').strip()
                    break
                elif len(next_line.split()) > 10:  # This looks like actual speech content
                    # Assume this is an answer if we don't have clear markers
                    entry_type = 'answer' if current_question_num > 0 else 'question'
                    if entry_type == 'question':
                        question_num += 1
                        current_question_num = question_num
                    speaker_details = "Unknown Role"
                    break
            
            if entry_type:
                # Collect the spoken text
                text_lines = []
                start_idx = i + 2 if speaker_details else i + 1
                
                for k in range(start_idx, len(lines)):
                    text_line = lines[k].strip()
                    
                    # Stop if we hit another potential speaker
                    if (len(text_line.split()) <= 4 and 
                        len(text_line) < 50 and
                        any(c.isalpha() for c in text_line) and
                        k + 1 < len(lines) and
                        lines[k + 1].strip()):
                        break
                        
                    if text_line and not re.match(r'^[.\-=_\s]{3,}$', text_line):
                        text_lines.append(text_line)
                
                if text_lines:
                    full_text = ' '.join(text_lines).strip()
                    full_text = re.sub(r'\.{3,}', '', full_text)
                    full_text = re.sub(r' {2,}', ' ', full_text)
                    
                    if len(full_text) > 20:
                        qa_data.append({
                            'question_number': current_question_num if current_question_num > 0 else 1,
                            'type': entry_type,
                            'speaker_name': speaker_name,
                            'speaker_details': speaker_details,
                            'text': full_text
                        })
                        
                        print(f"Alt method - {entry_type} from {speaker_name}: {full_text[:80]}...")
                
                i = start_idx + len(text_lines)
            else:
                i += 1
        else:
            i += 1
    
    print(f"Alternative method extracted: {len(qa_data)} entries")
    return qa_data

def simple_regex_extract(text: str) -> tuple:
    """Enhanced regex-based extraction that captures ALL questions and answers."""
    
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
    
    # Clean presentation thoroughly
    clean_presentation = clean_presentation_text(presentation)
    
    # Enhanced Q&A extraction - more aggressive approach
    qa_data = []
    if qa_section:
        print(f"Processing Q&A section with {len(qa_section)} characters...")
        
        # Remove operator comments from Q&A section
        qa_lines = []
        for line in qa_section.split('\n'):
            line_clean = line.strip()
            if not (line_clean.startswith('Operator:') or line_clean.startswith('Operator ')):
                qa_lines.append(line)
        
        qa_text = '\n'.join(qa_lines)
        
        # More flexible pattern matching - look for Q and A markers
        # Split by potential speaker boundaries
        sections = re.split(r'\n(?=[A-Za-z][A-Za-z\s,.-]{1,50}\n)', qa_text)
        
        question_num = 0
        current_question_num = 0
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if len(lines) < 2:
                continue
            
            # First line should be speaker name
            potential_name = lines[0].strip()
            
            # Second line should be title with Q or A
            if len(lines) > 1:
                potential_title = lines[1].strip()
                
                # Check for Q or A markers (more flexible)
                is_question = False
                is_answer = False
                
                if (potential_title.endswith(' Q') or 
                    potential_title.endswith('Q') or
                    'analyst' in potential_title.lower() or
                    'research' in potential_title.lower()):
                    is_question = True
                    question_num += 1
                    current_question_num = question_num
                    entry_type = 'question'
                    speaker_details = potential_title.replace(' Q', '').replace('Q', '').strip()
                    
                elif (potential_title.endswith(' A') or 
                      potential_title.endswith('A') or
                      any(title in potential_title.lower() for title in ['ceo', 'cfo', 'chief', 'president', 'officer', 'director'])):
                    is_answer = True
                    entry_type = 'answer'
                    speaker_details = potential_title.replace(' A', '').replace('A', '').strip()
                    
                if is_question or is_answer:
                    # Get the actual spoken text (everything after name and title)
                    text_content = '\n'.join(lines[2:]).strip()
                    
                    if text_content:
                        # Clean the text
                        text_content = re.sub(r'\.{3,}', '', text_content)  # Remove multiple dots
                        text_content = re.sub(r' {2,}', ' ', text_content)   # Remove multiple spaces
                        text_content = re.sub(r'\n+', ' ', text_content)     # Replace newlines with spaces
                        
                        if len(text_content) > 10:  # Only add if substantial content
                            qa_data.append({
                                'question_number': current_question_num if current_question_num > 0 else 1,
                                'type': entry_type,
                                'speaker_name': potential_name,
                                'speaker_details': speaker_details,
                                'text': text_content
                            })
                            
                            print(f"Extracted {entry_type} from {potential_name}: {text_content[:100]}...")
        
        print(f"Total extracted: {len(qa_data)} Q&A entries")
        
        # If we didn't get much, try alternative approach
        if len(qa_data) < 5:
            print("Trying alternative extraction method...")
            qa_data_alt = extract_qa_alternative_method(qa_section)
            if len(qa_data_alt) > len(qa_data):
                qa_data = qa_data_alt
    
    return clean_presentation, qa_data

def create_manual_prompts(text: str, output_dir: str):
    """Create manual prompts for use with any LLM."""
    
    qa_start = text.find("QUESTION AND ANSWER SECTION")
    if qa_start == -1:
        qa_start = text.find("Q&A")
    
    qa_section = text[qa_start:] if qa_start != -1 else "No Q&A section found"
    
    # Remove operator comments from qa_section
    qa_lines = []
    for line in qa_section.split('\n'):
        if not (line.strip().startswith('Operator:') or line.strip().startswith('Operator ')):
            qa_lines.append(line)
    qa_section = '\n'.join(qa_lines)
    
    prompt = """
Extract ALL Q&A data from this earnings call transcript. Multiple people may answer the same question.

Return as JSON array with ALL questions and answers:

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
  },
  {
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Bob Smith",
    "speaker_details": "CFO", 
    "text": "To add to that..."
  },
  {
    "question_number": 2,
    "type": "question",
    "speaker_name": "Mary Johnson",
    "speaker_details": "Analyst, ABC Bank",
    "text": "My next question..."
  }
]

Rules:
- Extract EVERY question and EVERY answer
- Multiple people can answer same question (keep same question_number)
- Increment question_number only for NEW questions
- Ignore ALL Operator comments completely
- Return only JSON

TRANSCRIPT:
[PASTE THE Q&A SECTION HERE]
"""
    
    # Save prompt and Q&A section
    with open(f"{output_dir}/manual_prompt.txt", 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    with open(f"{output_dir}/qa_section.txt", 'w', encoding='utf-8') as f:
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
    with open(f"{output_dir}/presentation.txt", 'w', encoding='utf-8') as f:
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

# SINGLE MAIN BLOCK - NO DUPLICATES
if __name__ == "__main__":
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