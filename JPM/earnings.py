#!/usr/bin/env python3
"""
Earnings Call Transcript Processor

A comprehensive tool for processing earnings call transcripts from PDF or DOCX files.
Extracts presentation content and Q&A sections, then uses OpenAI's GPT model to 
parse and structure the data for analysis.

Date: 2025-06-14

Dependencies:
    - openai: For GPT API calls
    - pandas: For data manipulation and export
    - PyPDF2: For PDF processing (optional)
    - python-docx: For DOCX processing (optional)

Outputs:
    - presentation.txt: Cleaned presentation content
    - qa_data.csv: Structured Q&A data in CSV format
    - qa_data.xlsx: Structured Q&A data in Excel format
    - Debug files: prompt_chunk_*.txt and raw_response_chunk_*.txt
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from openai import OpenAI

# Optional imports with graceful degradation
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


class TranscriptProcessor:
    """
    Main class for processing earnings call transcripts.
    
    This class handles the entire pipeline from file reading to structured output,
    including text cleaning, Q&A extraction, and LLM-based parsing.
    """
    
    # Class constants
    DEFAULT_CHUNK_SIZE = 7000
    QA_CHUNK_SIZE = 8000
    MAX_TOKENS = 1500
    TEMPERATURE = 0
    MODEL_NAME = "gpt-4-turbo"
    
    def __init__(self, ticker: str, year: int, quarter: int, api_key: Optional[str] = None):
        """
        Initialize the TranscriptProcessor.
        
        Args:
            ticker: company ticker code
            year: year
            quarter: number of the quarter between 1 and 4
            api_key (Optional[str]): OpenAI API key. If None, will attempt to 
                                   retrieve from environment variable.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.ticker = ticker.upper()
        self.year = year
        self.quarter = quarter
        output_file = f"{ticker}_{year}_Q{quarter}"
        self.output_presentation_file = f"{output_file}_presentation.txt"
        self.output_qa_file = f"{output_file}_qa_data.xlsx"
        # Create output directory if it doesn't exist
        if not os.path.exists("output"):
            os.makedirs("output")
        self.output_dir = Path("output")
        self.output_presentation_path = os.path.join(self.output_dir, self.output_presentation_file)
        self.output_qa_path = os.path.join(self.output_dir, self.output_qa_file)
        # delete and create log directory
        if os.path.exists("logs"):
            for file in os.listdir("logs"):
                os.remove(os.path.join("logs", file))
        self.log_dir = Path("logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def _fix_pdf_encoding(self, text: str) -> str:
        """
        Clean and fix common PDF encoding issues.
        
        PDFs often contain malformed Unicode characters that need to be replaced
        with their proper equivalents for better text processing.
        
        Args:
            text (str): Raw text extracted from PDF
            
        Returns:
            str: Cleaned text with fixed encoding issues
        """
        # Common PDF encoding fixes
        encoding_fixes = {
            'â€"': '—',     # Em dash
            'â€™': "'",     # Right single quotation mark
            'â€œ': '"',     # Left double quotation mark
            'â€': '"',      # Right double quotation mark
            'â€¢': '•',     # Bullet point
            'â€¦': '…',     # Horizontal ellipsis
            'â€˜': "'",     # Left single quotation mark
            'Â': '',        # Non-breaking space artifact
            'â€‹': '',      # Zero-width space
            'ï¿½': '',      # Replacement character
            'â€Š': ' ',     # Hair space
            'â€‰': ' ',     # Thin space
            'â€ˆ': ' ',     # Punctuation space
            'â€‡': ' ',     # Figure space
        }
        
        # Apply encoding fixes
        for malformed, correct in encoding_fixes.items():
            text = text.replace(malformed, correct)
        
        # Remove any remaining non-ASCII characters and normalize whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()
    
    def _read_pdf_file(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted and cleaned text content
            
        Raises:
            ImportError: If PyPDF2 is not installed
            Exception: If PDF reading fails
        """
        if not PyPDF2:
            raise ImportError(
                "PyPDF2 is required to process PDF files. "
                "Install it with: pip install PyPDF2"
            )
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_pages = []
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty pages
                        text_pages.append(page_text)
                
                raw_text = '\n'.join(text_pages)
                return self._fix_pdf_encoding(raw_text)
                
        except Exception as e:
            raise Exception(f"Failed to read PDF file '{file_path}': {str(e)}")
    
    def _read_docx_file(self, file_path: str) -> str:
        """
        Extract text content from a DOCX file.
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            str: Extracted text content
            
        Raises:
            ImportError: If python-docx is not installed
            Exception: If DOCX reading fails
        """
        if not docx:
            raise ImportError(
                "python-docx is required to process DOCX files. "
                "Install it with: pip install python-docx"
            )
        
        try:
            document = docx.Document(file_path)
            paragraphs = []
            
            for paragraph in document.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    paragraphs.append(paragraph.text)
            
            return '\n'.join(paragraphs)
            
        except Exception as e:
            raise Exception(f"Failed to read DOCX file '{file_path}': {str(e)}")
    
    def read_file(self, file_path: str) -> str:
        """
        Read and extract text from supported file formats.
        
        Supports PDF and DOCX files with automatic format detection based on
        file extension.
        
        Args:
            file_path (str): Path to the input file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            Exception: If file format is unsupported or reading fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._read_pdf_file(file_path)
        
        if file_extension in ['.docx', '.doc']:
            return self._read_docx_file(file_path)

        raise Exception(
            f"Unsupported file format: {file_extension}. "
            "Supported formats: .pdf, .docx, .doc"
        )
    
    def _find_qa_section_start(self, text: str) -> int:
        """
        Locate the start of the Q&A section in the transcript.
        
        Uses multiple patterns to identify where the Q&A section begins,
        as different transcripts may use various formatting styles.
        
        Args:
            text (str): Full transcript text
            
        Returns:
            int: Character position where Q&A section starts, or -1 if not found
        """
        qa_patterns = [
            r'QUESTION AND ANSWER SECTION',
            r'QUESTIONS? AND ANSWERS?',
            r'Q&A SECTION',
            r'Q&A',
            r'(?i)OPERATOR:.*(?:question|Q:)',
            r'(?i)Q:',  # Simple Q: pattern as fallback
        ]
        
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start()
        
        return -1
    
    def _create_presentation_cleaning_prompt(self, chunk: str) -> str:
        """
        Create a prompt for cleaning presentation content using LLM.
        
        Args:
            chunk (str): Text chunk to be cleaned
            
        Returns:
            str: Formatted prompt for the LLM
        """
        return f"""
The following is a section from the presentation portion of an earnings call transcript.

Your task:
- Remove all lines or phrases spoken by the conference operator.
  Examples include: "Your line is open.", "We'll now take our next question.", "Thank you.", "Please hold while we connect your call.", etc.
- Keep only the spoken remarks from company executives and analysts.
- Do not add any headers like "Cleaned Presentation:" or introductory text.
- Do not summarize or rephrase anything. Retain the exact wording and paragraph flow.
- Preserve paragraph structure unless it causes fragmentation — in that case, merge into the previous paragraph naturally.
- Remove any technical artifacts like page numbers or formatting remnants.

Return only the cleaned transcript text, with no extra commentary.

Transcript:
{chunk}"""
    
    def _clean_presentation_with_llm(self, text: str) -> str:
        """
        Clean presentation text using LLM processing in chunks.
        
        Processes the presentation text in manageable chunks to remove operator
        instructions and other non-content elements while preserving the actual
        presentation content.
        
        Args:
            text (str): Raw presentation text
            
        Returns:
            str: Cleaned presentation text
        """
        # Split text into manageable chunks
        chunks = [
            text[i:i + self.DEFAULT_CHUNK_SIZE] 
            for i in range(0, len(text), self.DEFAULT_CHUNK_SIZE)
        ]
        
        cleaned_chunks = []
        
        for idx, chunk in enumerate(chunks, 1):
            prompt = self._create_presentation_cleaning_prompt(chunk)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.MAX_TOKENS,
                    temperature=self.TEMPERATURE,
                )
                
                cleaned_text = response.choices[0].message.content.strip()
                cleaned_chunks.append(cleaned_text)
                
                # Save debug files
                debug_file = f"cleaned_presentation_prompt_chunk_{idx}.txt"
                self._save_debug_file(debug_file, prompt)
                debug_file = f"cleaned_presentation_response_chunk_{idx}.txt"
                self._save_debug_file(debug_file, cleaned_text)
                
                print(f"✅ Presentation chunk {idx}/{len(chunks)} cleaned successfully")
                
            except Exception as e:
                print(f"❌ Error cleaning presentation chunk {idx}: {e}")
                # Continue with other chunks even if one fails
                continue
        
        return "\n\n".join(cleaned_chunks)
    
    def _create_qa_extraction_prompt(self, chunk: str) -> str:
        """
        Create a prompt for extracting Q&A data using LLM.
        
        Args:
            chunk (str): Q&A text chunk to be processed
            
        Returns:
            str: Formatted prompt for the LLM
        """
        return f"""
From the following earnings call transcript, extract all questions and answers.

Instructions:
- Remove any lines spoken by the conference operator (e.g., "Your line is open", "We'll take our next question", "Please hold", etc.).
- Group each Q&A pair together sequentially.
- Identify the questioner and responder by name or role if available.
- If speaker identification is unclear, use generic labels like "Analyst" or "Executive".
- Preserve the exact wording of questions and answers.
- Format the output as a valid JSON array with the structure shown below.

Required JSON format:
[
  {{
    "question_number": 1,
    "type": "question",
    "speaker_name": "John Smith",
    "speaker_details": "Analyst, Goldman Sachs",
    "text": "My question is about..."
  }},
  {{
    "question_number": 1,
    "type": "answer",
    "speaker_name": "Jane Doe",
    "speaker_details": "CEO",
    "text": "Thanks for your question..."
  }}
]

Transcript:
{chunk}"""
    
    def _process_qa_chunk(self, chunk: str, chunk_idx: int) -> List[Dict]:
        """
        Process a single Q&A chunk using LLM.
        
        Args:
            chunk (str): Q&A text chunk
            chunk_idx (int): Index of the current chunk
            
        Returns:
            List[Dict]: Extracted Q&A entries for this chunk
        """
        prompt = self._create_qa_extraction_prompt(chunk)
        
        # Save prompt for debugging
        debug_file = f"qa_prompt_chunk_{chunk_idx}.txt"
        self._save_debug_file(debug_file, prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Save raw response for debugging
            debug_file = f"qa_response_chunk_{chunk_idx}.txt"
            self._save_debug_file(debug_file, raw_response)
            
            if not raw_response:
                print(f"⚠️  Chunk {chunk_idx} returned empty response")
                return []
            
            # Extract JSON from response
            if "[" in raw_response and "]" in raw_response:
                json_start = raw_response.find("[")
                json_end = raw_response.rfind("]") + 1
                json_str = raw_response[json_start:json_end]
                
                try:
                    chunk_data = json.loads(json_str)
                    if isinstance(chunk_data, list):
                        print(f"✅ Chunk {chunk_idx} processed: {len(chunk_data)} entries extracted")
                        return chunk_data

                    print(f"⚠️  Chunk {chunk_idx}: JSON structure is not a list")
                    return []
                        
                except json.JSONDecodeError as json_err:
                    print(f"❌ Chunk {chunk_idx} JSON parsing failed: {json_err}")
                    debug_file = f"error_chunk_{chunk_idx}.log"
                    self._save_debug_file(
                        debug_file, 
                        f"JSON parsing error: {json_err}\n\nRaw content:\n{raw_response}"
                    )
                    return []
            else:
                print(f"⚠️  Chunk {chunk_idx}: No valid JSON array found in response")
                debug_file = f"error_chunk_{chunk_idx}.log"
                self._save_debug_file(
                    debug_file, 
                    f"Missing JSON brackets\n\nRaw content:\n{raw_response}"
                )
                return []
                
        except Exception as e:
            print(f"❌ Chunk {chunk_idx} processing failed: {e}")
            debug_file = f"error_chunk_{chunk_idx}.log"
            self._save_debug_file(
                debug_file, 
                f"Processing error: {e}"
            )
            return []
    
    def _extract_qa_with_llm(self, qa_text: str) -> List[Dict]:
        """
        Extract Q&A data from text using LLM processing.
        
        Processes Q&A section in chunks and combines results into a structured
        format suitable for analysis.
        
        Args:
            qa_text (str): Q&A section text
            
        Returns:
            List[Dict]: Structured Q&A data with questions and answers
        """
        if not qa_text or len(qa_text.strip()) < 100:
            print("⚠️  Q&A section too short or empty, skipping LLM extraction")
            return []
        
        # Split Q&A into chunks
        qa_chunks = [
            qa_text[i:i + self.QA_CHUNK_SIZE] 
            for i in range(0, len(qa_text), self.QA_CHUNK_SIZE)
        ]
        
        print(f"📊 Processing {len(qa_chunks)} Q&A chunks")
        
        all_qa_data = []
        
        for idx, chunk in enumerate(qa_chunks, 1):
            print(f"🔄 Processing Q&A chunk {idx}/{len(qa_chunks)}")
            chunk_data = self._process_qa_chunk(chunk, idx)
            all_qa_data.extend(chunk_data)
        
        # Renumber questions sequentially
        return self._renumber_questions(all_qa_data)
    
    def _renumber_questions(self, qa_data: List[Dict]) -> List[Dict]:
        """
        Renumber questions sequentially across all chunks.
        
        Since chunks are processed independently, question numbers may not be
        sequential. This method ensures proper numbering.
        
        Args:
            qa_data (List[Dict]): Raw Q&A data with potentially non-sequential numbering
            
        Returns:
            List[Dict]: Q&A data with properly sequential question numbers
        """
        renumbered_data = []
        current_question_number = 1
        
        for entry in qa_data:
            if entry.get('type') == 'question':
                # Start of new Q&A pair
                question_number_for_pair = current_question_number
                current_question_number += 1
            
            # Update the question number for both questions and answers
            entry['question_number'] = question_number_for_pair
            renumbered_data.append(entry)
        
        return renumbered_data
    
    def _save_debug_file(self, debug_file: str, content: str) -> None:
        """
        Save debug content to file with error handling.
        
        Args:
            debug_file (str): file name to save
            content (str): Content to save
        """
        debug_path = os.path.join(self.log_dir, debug_file)
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Saved: {debug_path}")
        except Exception as e:
            print(f"⚠️  Failed to save debug file {debug_path}: {e}")
    
    def extract_content(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Extract and process both presentation and Q&A content from transcript.
        
        This is the main processing method that coordinates the extraction of
        presentation content and Q&A data using LLM processing.
        
        Args:
            text (str): Full transcript text
            
        Returns:
            Tuple[str, List[Dict]]: Cleaned presentation text and structured Q&A data
        """
        print("🔍 Locating Q&A section...")
        qa_start_position = self._find_qa_section_start(text)
        
        if qa_start_position != -1:
            presentation_text = text[:qa_start_position]
            qa_text = text[qa_start_position:]
            print(f"✅ Found Q&A section at position {qa_start_position}")
        else:
            presentation_text = text
            qa_text = ""
            print("⚠️  Q&A section not found, treating entire text as presentation")
        
        print("🧹 Cleaning presentation content...")
        cleaned_presentation = self._clean_presentation_with_llm(presentation_text)
        
        print("❓ Extracting Q&A data...")
        qa_data = self._extract_qa_with_llm(qa_text)
        
        print(f"✅ Extraction complete: {len(qa_data)} Q&A entries processed")
        return cleaned_presentation, qa_data
    
    def _save_outputs(self, presentation: str, qa_data: List[Dict]) -> None:
        """
        Save processed content to output files.
        
        Args:
            presentation (str): Cleaned presentation text
            qa_data (List[Dict]): Structured Q&A data
        """
        
        # Save presentation text
        try:
            with open(self.output_presentation_path, 'w', encoding='utf-8') as f:
                f.write(presentation)
            print(f"✅ Saved: {self.output_presentation_path}")
        except Exception as e:
            print(f"❌ Failed to save {self.output_presentation_path}: {e}")
        
        # Save Q&A data as DataFrame and export to multiple formats
        if qa_data:
            df = pd.DataFrame(qa_data)
            
            # Save as Excel
            try:
                df.to_excel(self.output_qa_path, index=False)
                print(f"✅ Saved: {self.output_qa_path}")
            except Exception as e:
                print(f"❌ Failed to save {self.output_qa_path}: {e}")
        else:
            print("⚠️  No Q&A data to save")
    
    def process_transcript(self, file_path: str) -> Tuple[str, pd.DataFrame]:
        """
        Complete transcript processing pipeline.
        
        This is the main public method that orchestrates the entire processing
        workflow from file reading to output generation.
        
        Args:
            file_path (str): Path to the transcript file (PDF or DOCX)
            
        Returns:
            Tuple[str, pd.DataFrame]: Processed presentation text and Q&A DataFrame
            
        Raises:
            Exception: If any step in the processing pipeline fails
        """
        print(f"📖 Reading transcript from: {file_path}")
        
        # Read and extract text from file
        raw_text = self.read_file(file_path)
        print(f"📄 Extracted {len(raw_text):,} characters from transcript")
        
        # Process content using LLM
        presentation, qa_data = self.extract_content(raw_text)
        
        # Save outputs
        self._save_outputs(presentation, qa_data)
        
        # Return processed data
        qa_dataframe = pd.DataFrame(qa_data) if qa_data else pd.DataFrame()
        
        print("🎉 Processing completed successfully!")
        print(f"📊 Results: {len(presentation.split())} words in presentation, {len(qa_data)} Q&A entries")
        
        return presentation, qa_dataframe


def main() -> None:
    """
    Main function to run the transcript processor.
    
    Demonstrates usage of the TranscriptProcessor class with a specific file.
    In production, this could be modified to accept command-line arguments.
    """
    start_time = time.time()
    try:
        # Configuration - modify these values as needed
        file_name = "JPM_1q25-earnings-transcript.pdf"
        file_directory = "JPM/JPM Presentation texts/2025/Q1"
        file_path = os.path.join(file_directory, file_name)
        ticker = "JPM"
        year = 2025
        quarter = 1

        #file_name = "250429-1q-2025-earnings-release-investors-and-analysts-call-transcript.pdf"
        #file_directory = "HSBC/HSBC Presentation texts/2025"
        #file_path = os.path.join(file_directory, file_name)
        #ticker = "HSBC"
        #year = 2025
        #quarter = 1
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        
        # Initialize processor and run
        processor = TranscriptProcessor(ticker, year, quarter)
        presentation, qa_dataframe = processor.process_transcript(file_path)
        
        # Display summary statistics
        print("\n" + "="*60)
        print("📈 PROCESSING SUMMARY")
        print("="*60)
        print(f"Input file: {file_path}")
        print(f"Output directory: {processor.output_dir}")
        print(f"Presentation words: {len(presentation.split()):,}")
        print(f"Q&A entries: {len(qa_dataframe):,}")
        if not qa_dataframe.empty:
            questions = len(qa_dataframe[qa_dataframe['type'] == 'question'])
            answers = len(qa_dataframe[qa_dataframe['type'] == 'answer'])
            print(f"Questions: {questions}, Answers: {answers}")
        
        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":

    print(f"🚀 Starting transcript processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"🏁 Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")