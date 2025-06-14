import pandas as pd
import json
import re, os, sys
from typing import Dict, List, Tuple
import openai
from openai import OpenAI
from pathlib import Path

class TranscriptExtractor:
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
    
    def extract_presentation_text(self, transcript: str) -> str:
        """Use LLM to extract and clean presentation text with chunking."""
        
        # First, try to find and extract just the presentation section
        qa_start = transcript.find("QUESTION AND ANSWER SECTION")
        if qa_start != -1:
            presentation_section = transcript[:qa_start]
        else:
            presentation_section = transcript
        
        # Check if still too large, chunk if needed
        max_chars = 12000  # Conservative limit for GPT-4
        
        if len(presentation_section) <= max_chars:
            return self._extract_presentation_chunk(presentation_section)
        else:
            return self._extract_presentation_in_chunks(presentation_section, max_chars)
    
    def _extract_presentation_chunk(self, text: str) -> str:
        """Extract presentation from a single chunk."""
        
        presentation_prompt = """
        You are tasked with extracting the presentation portion of an earnings call transcript. 
        
        Instructions:
        1. Extract ONLY the management presentation/discussion section
        2. Remove ALL operator comments (anything said by "Operator:")
        3. Remove separator lines (lines with many dots, dashes, or equals signs)
        4. Stop when you reach the "QUESTION AND ANSWER SECTION"
        5. Keep all executive commentary and financial results discussion
        6. Maintain the speaker attributions for executives (Jeremy Barnum, Jamie Dimon, etc.)
        7. Clean up formatting but preserve paragraph structure
        
        Return only the cleaned presentation text, nothing else.
        
        Transcript:
        {transcript}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5 for better rate limits
                messages=[
                    {"role": "system", "content": "You are a financial transcript processing expert."},
                    {"role": "user", "content": presentation_prompt.format(transcript=text)}
                ],
                max_tokens=3000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error extracting presentation chunk: {e}")
            return ""
    
    def _extract_presentation_in_chunks(self, text: str, chunk_size: int) -> str:
        """Extract presentation by processing in chunks."""
        
        print("Presentation section is large, processing in chunks...")
        
        # Split into logical chunks at paragraph breaks
        chunks = self._split_text_smart(text, chunk_size)
        
        all_presentation_parts = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing presentation chunk {i+1}/{len(chunks)}...")
            
            # Add rate limiting delay
            if i > 0:
                import time
                time.sleep(2)  # 2 second delay between requests
            
            chunk_result = self._extract_presentation_chunk(chunk)
            if chunk_result.strip():
                all_presentation_parts.append(chunk_result)
        
        return '\n\n'.join(all_presentation_parts)
    
    def extract_qa_data(self, transcript: str) -> pd.DataFrame:
        """Use LLM to extract and structure Q&A data with chunking."""
        
        # Find Q&A section
        qa_start = transcript.find("QUESTION AND ANSWER SECTION")
        if qa_start == -1:
            print("No Q&A section found")
            return pd.DataFrame()
        
        qa_section = transcript[qa_start:]
        
        # Process in chunks due to size limits
        return self._extract_qa_in_chunks(qa_section)
    
    def _extract_qa_in_chunks(self, qa_text: str) -> pd.DataFrame:
        """Extract Q&A data by processing in chunks."""
        
        max_chars = 10000  # Conservative limit
        chunks = self._split_qa_into_logical_chunks(qa_text, max_chars)
        
        all_qa_data = []
        question_offset = 0
        
        for i, chunk in enumerate(chunks):
            print(f"Processing Q&A chunk {i+1}/{len(chunks)}...")
            
            # Add rate limiting delay
            if i > 0:
                import time
                time.sleep(3)  # 3 second delay between requests
            
            chunk_df = self._extract_qa_chunk(chunk)
            
            if not chunk_df.empty:
                # Adjust question numbers to maintain continuity
                if question_offset > 0:
                    max_q_num_in_chunk = chunk_df['question_number'].max()
                    chunk_df['question_number'] = chunk_df['question_number'] + question_offset
                    question_offset += max_q_num_in_chunk
                else:
                    question_offset = chunk_df['question_number'].max()
                
                all_qa_data.append(chunk_df)
        
        if all_qa_data:
            return pd.concat(all_qa_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _extract_qa_chunk(self, qa_text: str) -> pd.DataFrame:
        """Extract Q&A data from a single chunk with robust JSON parsing."""
        
        qa_extraction_prompt = """
        Extract Q&A data from this earnings call transcript chunk. Return ONLY a valid JSON array.

        Rules:
        1. Find speaker interactions (ignore Operator comments)
        2. For each speaker, determine:
           - Speaker name
           - Title/company 
           - Whether they're asking (question) or answering (answer)
           - The text they spoke
        3. Number questions sequentially starting from 1
        4. Use same question_number for the answer that follows

        Return EXACTLY this JSON format with no extra text:
        [
          {
            "question_number": 1,
            "type": "question",
            "speaker_name": "Ken Usdin",
            "speaker_details": "Analyst, Autonomous Research",
            "text": "Good morning, Jeremy..."
          },
          {
            "question_number": 1,
            "type": "answer",
            "speaker_name": "Jeremy Barnum", 
            "speaker_details": "Chief Financial Officer, JPMorganChase",
            "text": "Sure, Ken..."
          }
        ]

        Transcript:
        {transcript}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract Q&A data and return ONLY valid JSON arrays. No explanations, no markdown formatting."},
                    {"role": "user", "content": qa_extraction_prompt.format(transcript=qa_text)}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Get the raw response
            raw_response = response.choices[0].message.content.strip()
            
            # Clean and parse JSON with multiple strategies
            qa_data = self._parse_json_response(raw_response)
            
            if qa_data is None:
                print(f"Failed to parse JSON response")
                return pd.DataFrame()
            
            # Validate the structure
            if not isinstance(qa_data, list):
                print(f"Response is not a list: {type(qa_data)}")
                return pd.DataFrame()
            
            # Filter out invalid entries
            valid_entries = []
            for entry in qa_data:
                if self._validate_qa_entry(entry):
                    valid_entries.append(entry)
                else:
                    print(f"Invalid entry skipped: {entry}")
            
            return pd.DataFrame(valid_entries)
            
        except Exception as e:
            print(f"Error in Q&A chunk extraction: {e}")
            return pd.DataFrame()
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Parse JSON response with multiple fallback strategies."""
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove markdown formatting
        cleaned = response
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Find JSON array in response
        import re
        json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Fix common JSON issues
        try:
            # Fix trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            # Fix missing quotes around keys
            fixed = re.sub(r'(\w+):', r'"\1":', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Parse manually if all else fails
        return self._manual_parse_qa_response(response)
    
    def _manual_parse_qa_response(self, response: str) -> List[Dict]:
        """Manually parse Q&A response when JSON parsing fails."""
        
        try:
            # Look for patterns in the response
            entries = []
            lines = response.split('\n')
            
            current_entry = {}
            for line in lines:
                line = line.strip()
                
                # Look for key-value patterns
                if '"question_number"' in line and ':' in line:
                    try:
                        value = int(re.search(r':\s*(\d+)', line).group(1))
                        current_entry['question_number'] = value
                    except:
                        current_entry['question_number'] = 1
                
                elif '"type"' in line and ':' in line:
                    if 'question' in line.lower():
                        current_entry['type'] = 'question'
                    elif 'answer' in line.lower():
                        current_entry['type'] = 'answer'
                
                elif '"speaker_name"' in line and ':' in line:
                    try:
                        value = re.search(r':\s*"([^"]+)"', line).group(1)
                        current_entry['speaker_name'] = value
                    except:
                        current_entry['speaker_name'] = 'Unknown'
                
                elif '"speaker_details"' in line and ':' in line:
                    try:
                        value = re.search(r':\s*"([^"]+)"', line).group(1)
                        current_entry['speaker_details'] = value
                    except:
                        current_entry['speaker_details'] = ''
                
                elif '"text"' in line and ':' in line:
                    try:
                        value = re.search(r':\s*"(.+)"', line).group(1)
                        current_entry['text'] = value
                        
                        # Entry is complete
                        if len(current_entry) >= 4:
                            entries.append(current_entry.copy())
                            current_entry = {}
                    except:
                        pass
            
            return entries if entries else []
            
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return []
    
    def _validate_qa_entry(self, entry: Dict) -> bool:
        """Validate a Q&A entry has required fields."""
        
        required_fields = ['question_number', 'type', 'speaker_name', 'speaker_details', 'text']
        
        if not isinstance(entry, dict):
            return False
        
        for field in required_fields:
            if field not in entry:
                return False
            
            # Check for empty or invalid values
            if field == 'question_number':
                if not isinstance(entry[field], (int, float)) or entry[field] < 1:
                    return False
            elif field == 'type':
                if entry[field] not in ['question', 'answer']:
                    return False
            elif field in ['speaker_name', 'text']:
                if not entry[field] or not entry[field].strip():
                    return False
        
        return True
    
    def _split_text_smart(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at logical boundaries."""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                            if current_chunk:
                                current_chunk += '. ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_qa_into_logical_chunks(self, qa_text: str, chunk_size: int) -> List[str]:
        """Split Q&A text into chunks at speaker boundaries."""
        
        if len(qa_text) <= chunk_size:
            return [qa_text]
        
        chunks = []
        current_chunk = ""
        
        # Split by double newlines to preserve speaker sections
        sections = qa_text.split('\n\n')
        
        for section in sections:
            if len(current_chunk) + len(section) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + section
                else:
                    current_chunk = section
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = section
                else:
                    # Section too long, split further
                    lines = section.split('\n')
                    for line in lines:
                        if len(current_chunk) + len(line) + 1 <= chunk_size:
                            if current_chunk:
                                current_chunk += '\n' + line
                            else:
                                current_chunk = line
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_in_chunks(self, transcript: str, chunk_size: int = 15000) -> pd.DataFrame:
        """Process large transcripts in chunks for Q&A extraction."""
        
        # This method is now redundant since extract_qa_data handles chunking
        return self.extract_qa_data(transcript)

def add_rate_limiting_delay(request_count: int):
    """Add appropriate delay based on rate limits."""
    import time
    
    if request_count > 0:
        # Conservative delay to avoid rate limits
        delay = min(3.0, request_count * 0.5)  # Increase delay with more requests
        print(f"Rate limiting: waiting {delay} seconds...")
        time.sleep(delay)
    
    def _split_qa_into_chunks(self, qa_text: str, chunk_size: int) -> List[str]:
        """Split Q&A text into chunks at speaker boundaries."""
        
        chunks = []
        current_chunk = ""
        
        # Split by speaker patterns to avoid breaking mid-conversation
        speaker_pattern = r'\n([A-Z][a-zA-Z\s]+)\n([A-Za-z,\s]+)\s+[QA]\n'
        parts = re.split(speaker_pattern, qa_text)
        
        for part in parts:
            if len(current_chunk) + len(part) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

def process_transcript_with_llm(transcript_text: str, openai_api_key: str, output_dir: str = "."):
    """
    Main function to process transcript using LLM prompts with improved error handling.
    
    Args:
        transcript_text: The full transcript text
        openai_api_key: Your OpenAI API key
        output_dir: Directory to save output files
    """
    
    extractor = TranscriptExtractor(openai_api_key)
    
    print("Extracting presentation text...")
    try:
        presentation_text = extractor.extract_presentation_text(transcript_text)
    except Exception as e:
        print(f"Error in presentation extraction: {e}")
        print("Falling back to manual extraction...")
        
        # Fallback: extract presentation manually
        qa_start = transcript_text.find("QUESTION AND ANSWER SECTION")
        if qa_start != -1:
            presentation_text = transcript_text[:qa_start]
        else:
            presentation_text = transcript_text
        
        # Basic cleanup
        lines = presentation_text.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('Operator:')]
        presentation_text = '\n'.join(cleaned_lines)
    
    print("Extracting Q&A data...")
    try:
        qa_dataframe = extractor.extract_qa_data(transcript_text)
    except Exception as e:
        print(f"Error in Q&A extraction: {e}")
        qa_dataframe = pd.DataFrame()
    
    # If Q&A extraction failed, suggest manual approach
    if qa_dataframe.empty:
        print("\n❌ Q&A extraction failed. Generating manual prompts instead...")
        
        # Generate manual prompts
        process_with_manual_prompts_backup(transcript_text, output_dir)
        
        # Create empty DataFrame for consistency
        qa_dataframe = pd.DataFrame(columns=['question_number', 'type', 'speaker_name', 'speaker_details', 'text'])
    
    # Save presentation text
    presentation_file = f"{output_dir}/presentation.txt"
    with open(presentation_file, 'w', encoding='utf-8') as f:
        f.write(presentation_text)
    
    # Save Q&A dataframe
    qa_csv_file = f"{output_dir}/qa_data.csv"
    qa_excel_file = f"{output_dir}/qa_data.xlsx"
    
    qa_dataframe.to_csv(qa_csv_file, index=False)
    qa_dataframe.to_excel(qa_excel_file, index=False)
    
    print(f"\nFiles created:")
    print(f"- Presentation: {presentation_file}")
    print(f"- Q&A CSV: {qa_csv_file}")
    print(f"- Q&A Excel: {qa_excel_file}")
    
    if not qa_dataframe.empty:
        print(f"\nQ&A Summary:")
        print(f"- Total entries: {len(qa_dataframe)}")
        print(f"- Questions: {len(qa_dataframe[qa_dataframe['type'] == 'question'])}")
        print(f"- Answers: {len(qa_dataframe[qa_dataframe['type'] == 'answer'])}")
        print(f"- Unique question numbers: {qa_dataframe['question_number'].nunique()}")
    else:
        print(f"\n⚠️  Q&A extraction failed. Manual prompts have been generated.")
        print(f"Check these files for manual processing:")
        print(f"- {output_dir}/qa_prompt_manual.txt")
        print(f"- {output_dir}/qa_section_extracted.txt")
    
    return presentation_text, qa_dataframe

def process_with_manual_prompts_backup(transcript_text: str, output_dir: str):
    """Generate manual prompts when automatic extraction fails."""
    
    # Extract Q&A section
    qa_start = transcript_text.find("QUESTION AND ANSWER SECTION")
    if qa_start != -1:
        qa_section = transcript_text[qa_start:]
    else:
        qa_section = "Q&A section not found in transcript"
    
    # Create simplified manual prompt
    manual_prompt = """
I need you to extract Q&A data from this earnings call transcript and return it as a JSON array.

For each speaker interaction:
1. Identify if it's a question (from analysts) or answer (from executives)
2. Extract speaker name and their title/company
3. Get their spoken text
4. Number questions sequentially (same number for question and its answer)

Return this exact JSON format:
[
  {
    "question_number": 1,
    "type": "question", 
    "speaker_name": "Ken Usdin",
    "speaker_details": "Analyst, Autonomous Research",
    "text": "Good morning, Jeremy..."
  },
  {
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Jeremy Barnum",
    "speaker_details": "Chief Financial Officer, JPMorganChase", 
    "text": "Sure, Ken..."
  }
]

TRANSCRIPT:
{qa_section}
"""
    
    # Save the manual prompt and Q&A section
    with open(f"{output_dir}/qa_prompt_manual.txt", 'w', encoding='utf-8') as f:
        f.write(manual_prompt.format(qa_section="[PASTE Q&A SECTION HERE]"))
    
    with open(f"{output_dir}/qa_section_extracted.txt", 'w', encoding='utf-8') as f:
        f.write(qa_section)
    
    print(f"Manual processing files created:")
    print(f"- {output_dir}/qa_prompt_manual.txt (copy this prompt)")
    print(f"- {output_dir}/qa_section_extracted.txt (copy this text)")
    print(f"\nTo use: Copy the prompt, replace [PASTE Q&A SECTION HERE] with the extracted text, then use in ChatGPT.")

# Alternative function for other LLM providers (Anthropic Claude, etc.)
def create_extraction_prompts():
    """
    Return the prompts that can be used with any LLM provider.
    Useful if you want to use Claude, Gemini, or other models.
    """
    
    presentation_prompt = """
    Extract the presentation portion of this earnings call transcript:
    
    RULES:
    1. Extract ONLY the management presentation/discussion section
    2. Remove ALL operator comments (anything said by "Operator:")
    3. Remove separator lines (lines with many dots, dashes, or equals signs)
    4. Stop when you reach the "QUESTION AND ANSWER SECTION"
    5. Keep all executive commentary and financial results discussion
    6. Maintain speaker attributions for executives
    7. Clean up formatting but preserve paragraph structure
    
    Return only the cleaned presentation text.
    
    TRANSCRIPT:
    {transcript}
    """
    
    qa_prompt = """
    Extract Q&A data from this earnings call transcript and return as JSON:
    
    RULES:
    1. Extract ONLY the Question & Answer section
    2. Ignore ALL operator comments
    3. For each interaction, identify: speaker name, title/company, text, type (question/answer)
    4. Question numbering: increment for each new analyst question, same number for corresponding answers
    5. Type: "question" for analysts, "answer" for executives
    
    JSON FORMAT:
    [
      {
        "question_number": 1,
        "type": "question",
        "speaker_name": "Speaker Name",
        "speaker_details": "Title, Company",
        "text": "What they said..."
      }
    ]
    
    TRANSCRIPT:
    {transcript}
    """
    
    return {
        "presentation_prompt": presentation_prompt,
        "qa_prompt": qa_prompt
    }

def read_input_file(file_path: str) -> str:
    """
    Read transcript from various file formats.
    
    Args:
        file_path: Path to the input file (supports .txt, .pdf, .docx)
    
    Returns:
        String content of the file
    """
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.txt':
            return read_text_file(file_path)
        elif file_extension == '.pdf':
            return read_pdf_file(file_path)
        elif file_extension in ['.docx', '.doc']:
            return read_word_file(file_path)
        else:
            # Try reading as text file
            print(f"Unknown file extension {file_extension}, attempting to read as text file...")
            return read_text_file(file_path)
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """Read a plain text file."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read text file with {encoding} encoding")
            return content
        except UnicodeDecodeError:
            continue
    
    raise Exception(f"Could not read file {file_path} with any supported encoding")

def read_pdf_file(file_path: str) -> str:
    """Read a PDF file and extract text."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        print(f"Successfully read PDF file: {len(text)} characters extracted")
        return text
    
    except ImportError:
        print("PyPDF2 not installed. Install with: pip install PyPDF2")
        raise
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise

def read_word_file(file_path: str) -> str:
    """Read a Word document (.docx or .doc)."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        print(f"Successfully read Word document: {len(text)} characters extracted")
        return text
    
    except ImportError:
        print("python-docx not installed. Install with: pip install python-docx")
        raise
    except Exception as e:
        print(f"Error reading Word document: {e}")
        raise

def setup_requirements():
    """Display installation instructions for optional dependencies."""
    print("Optional dependencies for different file formats:")
    print("For PDF files: pip install PyPDF2")
    print("For Word docs: pip install python-docx")
    print("For OpenAI API: pip install openai")

def process_transcript_file(file_path: str, openai_api_key: str, output_dir: str = "."):
    """
    Process transcript file using LLM prompts.
    
    Args:
        file_path: Path to input transcript file (.txt, .pdf, .docx)
        openai_api_key: Your OpenAI API key
        output_dir: Directory to save output files
    """
    
    print(f"Reading transcript from: {file_path}")
    transcript_text = read_input_file(file_path)
    
    if not transcript_text.strip():
        print("Error: No content extracted from file")
        return None, None
    
    print(f"Extracted {len(transcript_text)} characters from file")
    
    # Process with LLM
    return process_transcript_with_llm(transcript_text, openai_api_key, output_dir)

def process_with_manual_prompts(file_path: str, output_dir: str = "."):
    """
    Read file and provide prompts for manual use with any LLM.
    
    Args:
        file_path: Path to input transcript file
        output_dir: Directory to save the extracted text and prompt files
    """
    
    print(f"Reading transcript from: {file_path}")
    transcript_text = read_input_file(file_path)
    
    if not transcript_text.strip():
        print("Error: No content extracted from file")
        return
    
    print(f"Extracted {len(transcript_text)} characters from file")
    
    # Save the extracted text
    text_output_path = f"{output_dir}/extracted_transcript.txt"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    print(f"Saved extracted text to: {text_output_path}")
    
    # Get prompts
    prompts = create_extraction_prompts()
    
    # Save prompts to files for manual use
    presentation_prompt_file = f"{output_dir}/presentation_prompt.txt"
    qa_prompt_file = f"{output_dir}/qa_prompt.txt"
    
    with open(presentation_prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompts["presentation_prompt"].format(transcript="[PASTE YOUR TRANSCRIPT HERE]"))
    
    with open(qa_prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompts["qa_prompt"].format(transcript="[PASTE YOUR TRANSCRIPT HERE]"))
    
    print(f"\nPrompt files created:")
    print(f"- Presentation extraction: {presentation_prompt_file}")
    print(f"- Q&A extraction: {qa_prompt_file}")
    print("\nTo use manually:")
    print("1. Copy the prompt from the files")
    print("2. Replace '[PASTE YOUR TRANSCRIPT HERE]' with your actual transcript")
    print("3. Use with ChatGPT, Claude, or any other LLM")

# Example usage with different scenarios
if __name__ == "__main__":

    # Simple usage examples:
    
    # Automatic processing with API
    API_KEY = os.environ.get("OPENAI_API_KEY")  
    file_name = "JPM_1q25-earnings-transcript.pdf"
    file_dir = "JPM\\JPM presentation texts\\2025\\Q1"
    file_path = os.path.join(file_dir, file_name)
    output_dir = os.path.join("JPM", "output")
    presentation, qa_df = process_transcript_file(file_path, API_KEY, output_dir)
    