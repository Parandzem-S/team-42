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
        """Use LLM to extract and clean presentation text."""
        
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
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial transcript processing expert."},
                    {"role": "user", "content": presentation_prompt.format(transcript=transcript)}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error extracting presentation: {e}")
            sys.exit()
    
    def extract_qa_data(self, transcript: str) -> pd.DataFrame:
        """Use LLM to extract and structure Q&A data."""
        
        qa_extraction_prompt = """
        You are tasked with extracting Q&A data from an earnings call transcript.
        
        Instructions:
        1. Extract ONLY the Question & Answer section (after "QUESTION AND ANSWER SECTION")
        2. Ignore ALL operator comments
        3. For each speaker interaction, identify:
           - Speaker name (e.g., "Ken Usdin", "Jeremy Barnum")
           - Speaker details/title (e.g., "Analyst, Autonomous Research", "Chief Financial Officer, JPMorganChase")
           - The actual text they spoke
           - Whether it's a question (from analysts) or answer (from company executives)
        
        4. Question numbering rules:
           - Start question_number at 1 for the first analyst question
           - Increment question_number by 1 for each NEW analyst question
           - Keep the SAME question_number for the executive's answer to that question
           - If multiple executives answer the same question, they all get the same question_number
        
        5. Type classification:
           - "question" for analysts asking questions
           - "answer" for company executives responding
        
        Return the data as a JSON array with this exact structure:
        [
          {
            "question_number": 1,
            "type": "question",
            "speaker_name": "Ken Usdin",
            "speaker_details": "Analyst, Autonomous Research",
            "text": "Good morning, Jeremy. Wondering if you could start by..."
          },
          {
            "question_number": 1,
            "type": "answer", 
            "speaker_name": "Jeremy Barnum",
            "speaker_details": "Chief Financial Officer, JPMorganChase",
            "text": "Sure, Ken. So I mean, at a high level..."
          }
        ]
        
        Transcript:
        {transcript}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial transcript processing expert. Return only valid JSON."},
                    {"role": "user", "content": qa_extraction_prompt.format(transcript=transcript)}
                ],
                max_tokens=8000,
                temperature=0.1
            )
            
            # Parse JSON response
            json_response = response.choices[0].message.content.strip()
            
            # Clean up JSON if needed
            if json_response.startswith("```json"):
                json_response = json_response.replace("```json", "").replace("```", "").strip()
            
            qa_data = json.loads(json_response)
            
            # Convert to DataFrame
            return pd.DataFrame(qa_data)
        
        except Exception as e:
            print(f"Error extracting Q&A data: {e}")
            return pd.DataFrame()
    
    def process_in_chunks(self, transcript: str, chunk_size: int = 15000) -> pd.DataFrame:
        """Process large transcripts in chunks for Q&A extraction."""
        
        # Find Q&A section
        qa_start = transcript.find("QUESTION AND ANSWER SECTION")
        if qa_start == -1:
            print("No Q&A section found")
            return pd.DataFrame()
        
        qa_section = transcript[qa_start:]
        
        # If Q&A section is large, process in chunks
        if len(qa_section) <= chunk_size:
            return self.extract_qa_data(qa_section)
        
        # Split into chunks while preserving speaker boundaries
        chunks = self._split_qa_into_chunks(qa_section, chunk_size)
        
        all_qa_data = []
        question_offset = 0
        
        for i, chunk in enumerate(chunks):
            print(f"Processing Q&A chunk {i+1}/{len(chunks)}...")
            
            chunk_df = self.extract_qa_data(chunk)
            
            if not chunk_df.empty:
                # Adjust question numbers to maintain continuity
                if question_offset > 0:
                    chunk_df['question_number'] = chunk_df['question_number'] + question_offset
                
                all_qa_data.append(chunk_df)
                
                # Update offset for next chunk
                question_offset = chunk_df['question_number'].max()
        
        if all_qa_data:
            return pd.concat(all_qa_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
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
    Main function to process transcript using LLM prompts.
    
    Args:
        transcript_text: The full transcript text
        openai_api_key: Your OpenAI API key
        output_dir: Directory to save output files
    """
    
    extractor = TranscriptExtractor(openai_api_key)
    
    print("Extracting presentation text...")
    presentation_text = extractor.extract_presentation_text(transcript_text)
    
    print("Extracting Q&A data...")
    qa_dataframe = extractor.process_in_chunks(transcript_text)
    
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
    
    print(f"\nQ&A Summary:")
    print(f"- Total entries: {len(qa_dataframe)}")
    print(f"- Questions: {len(qa_dataframe[qa_dataframe['type'] == 'question'])}")
    print(f"- Answers: {len(qa_dataframe[qa_dataframe['type'] == 'answer'])}")
    print(f"- Unique question numbers: {qa_dataframe['question_number'].nunique()}")
    
    return presentation_text, qa_dataframe

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