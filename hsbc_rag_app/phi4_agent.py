# This file contains YOUR EXACT Phi-4 code from paste-2.txt

import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import gc
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class Phi4BankingAgent:
    """
    YOUR EXACT Phi-4 banking agent code from paste-2.txt
    """
    
    def __init__(self):
        """Initialize using YOUR EXACT setup code"""
        # Setup
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-mini-instruct",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cuda",
                low_cpu_mem_usage=True
            )
        else:
            print("WARNING: No GPU detected, using CPU (slow)")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-mini-instruct",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # YOUR EXACT generation args
        self.generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.3,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }

        self.batch_size = 16 if torch.cuda.is_available() else 2

    def create_prompt(self, question_text: str, answer_text: str) -> str:
        """
        YOUR EXACT create_prompt function
        """
        # Truncate to reasonable lengths
        q_truncated = question_text[:400] if len(question_text) > 400 else question_text
        a_truncated = answer_text[:500] if len(answer_text) > 500 else answer_text
        
        prompt = f"""Analyze this Q&A exchange and provide exactly the following format:

INSIGHT: [key insight from the answer in max 10 words - what was revealed, decided, or explained]
RISK: [Yes/No/Unclear]  
ANSWERED: [Complete/Partial/None]

Question: {q_truncated}

Answer: {a_truncated}

Analysis:"""
        
        return prompt

    def extract_summary(self, text: str) -> str:
        """YOUR EXACT extract_summary function"""
        # Look for INSIGHT: pattern
        patterns = [
            r"INSIGHT:\s*([^\n]+)",
            r"Insight:\s*([^\n]+)", 
            r"insight:\s*([^\n]+)",
            r"SUMMARY:\s*([^\n]+)",
            r"Summary:\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                insight = match.group(1).strip()
                # Limit to 10 words
                words = insight.split()
                if len(words) > 10:
                    insight = " ".join(words[:10])
                return insight
        
        # Fallback: extract first meaningful line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if not any(keyword in line.lower() for keyword in ['risk', 'answer', 'complete', 'partial', 'analysis']):
                words = line.split()
                if len(words) >= 3:
                    return " ".join(words[:10])
        
        return "No clear insight extracted"

    def extract_risk(self, text: str) -> str:
        """YOUR EXACT extract_risk function"""
        # Look for RISK: pattern from LLM response
        patterns = [
            r"RISK:\s*(Yes|No|Unclear)",
            r"Risk:\s*(Yes|No|Unclear)",
            r"risk:\s*(yes|no|unclear)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        # If no clear pattern found, return what the LLM likely intended
        # Look for Yes/No in the text without keyword assumptions
        if re.search(r'\byes\b', text, re.IGNORECASE):
            return "Yes"
        elif re.search(r'\bno\b', text, re.IGNORECASE):
            return "No"
        
        return "Unclear"

    def extract_coverage(self, text: str) -> str:
        """YOUR EXACT extract_coverage function"""
        # Look for ANSWERED: pattern from LLM response
        patterns = [
            r"ANSWERED:\s*(Complete|Partial|None)",
            r"Answered:\s*(Complete|Partial|None)",
            r"answered:\s*(complete|partial|none)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = match.group(1).capitalize()
                return "Fully answered" if result == "Complete" else f"{result}ly answered" if result == "Partial" else "Not answered"
        
        # If no clear pattern found, look for the exact terms the LLM might use
        if re.search(r'\bcomplete\b', text, re.IGNORECASE):
            return "Fully answered"
        elif re.search(r'\bpartial\b', text, re.IGNORECASE):
            return "Partially answered"
        elif re.search(r'\bnone\b', text, re.IGNORECASE):
            return "Not answered"
        
        return "Unclear"

    def analyze_qa_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        YOUR EXACT Q&A analysis logic from paste-2.txt
        """
        print("Extracting Q&A pairs based on Year, Quarter, Question No grouping (Q&A session only)...")

        # First, clean and prepare the data - YOUR CODE
        df['Question No'] = pd.to_numeric(df['Question No'], errors='coerce')

        # Filter for Q&A session only (presentation==0) and valid Question No - YOUR CODE
        qa_session_data = df[(df['presentation'] == 0) & (df['Question No'].notna())]

        # Group by Year, Quarter, Question No to get proper Q&A pairs - YOUR CODE
        qa_pairs = []

        # Get unique question groups from Q&A session only - YOUR CODE
        question_groups = qa_session_data.groupby(['Year', 'Quarter', 'Question No'])

        for (year, quarter, q_no), group in question_groups:
            # Sort by index to maintain order within each question group - YOUR CODE
            group_sorted = group.sort_index()
            
            # Separate questions and answers - YOUR CODE
            questions = group_sorted[group_sorted['flag_question'] == True]
            answers = group_sorted[group_sorted['flag_question'] == False]
            
            # Skip if no questions or answers - YOUR CODE
            if len(questions) == 0 or len(answers) == 0:
                continue
            
            # Combine all questions in this group - YOUR CODE
            combined_questions = " ".join(questions['Speaker text'].astype(str))
            question_indices = questions.index.tolist()
            
            # Combine all answers in this group - YOUR CODE
            combined_answers = " ".join(answers['Speaker text'].astype(str))
            answer_indices = answers.index.tolist()
            
            # Create Q&A pair - YOUR CODE
            qa_pairs.append({
                "questions": combined_questions,
                "answer": combined_answers,
                "question_indices": question_indices,
                "answer_indices": answer_indices,
                "year": year,
                "quarter": quarter,
                "question_no": q_no
            })

        print(f"Found {len(qa_pairs)} Q&A pairs from Q&A session only (presentation==0)")

        # Build prompts - YOUR CODE
        prompts = []
        for pair in qa_pairs:
            prompt = self.create_prompt(pair["questions"], pair["answer"])
            prompts.append(prompt)

        print(f"Created {len(prompts)} prompts for processing")

        # YOUR EXACT batch processing with error handling
        print("Processing batches...")
        outputs_all = []

        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Processing batches"):
                batch = prompts[i:i + self.batch_size]
                
                try:
                    outputs = self.pipe(batch, **self.generation_args)
                    batch_outputs = []
                    
                    for out in outputs:
                        if isinstance(out, list) and len(out) > 0:
                            batch_outputs.append(out[0]['generated_text'].strip())
                        else:
                            batch_outputs.append("Processing failed")
                            
                    outputs_all.extend(batch_outputs)
                    
                except Exception as e:
                    print(f"Batch {i//self.batch_size} failed: {e}")
                    # Process individually on failure
                    for prompt in batch:
                        try:
                            output = self.pipe(prompt, **self.generation_args)
                            if isinstance(output, list) and len(output) > 0:
                                outputs_all.append(output[0]['generated_text'].strip())
                            else:
                                outputs_all.append("Processing failed")
                        except Exception as e2:
                            print(f"Individual prompt failed: {e2}")
                            outputs_all.append("Processing failed")

                # Memory cleanup
                if i % (self.batch_size * 4) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # YOUR EXACT result processing
        # Initialize all columns with empty values - let LLM decide everything
        print("Initializing result columns...")
        df["key_findings"] = ""
        df["risk_or_distress"] = ""
        df["answer_coverage"] = ""

        # OPTIMIZED: Single loop for mapping and counting
        print("Mapping results to dataframe...")
        risk_detected_pairs = 0
        fully_answered_pairs = 0
        valid_insight_pairs = 0

        for i, (pair, llm_output) in enumerate(zip(qa_pairs, outputs_all)):
            if llm_output == "Processing failed":
                continue
                
            # Extract insights ONCE
            summary = self.extract_summary(llm_output)
            risk = self.extract_risk(llm_output)
            coverage = self.extract_coverage(llm_output)
            
            # Count pair statistics
            if risk == "Yes":
                risk_detected_pairs += 1
            if coverage == "Fully answered":
                fully_answered_pairs += 1
            if len(summary.split()) >= 3 and summary not in ["", "No clear insight extracted", "Processing failed"]:
                valid_insight_pairs += 1
            
            # Apply to ALL question rows in this group
            for q_idx in pair["question_indices"]:
                if q_idx < len(df):
                    df.loc[q_idx, "key_findings"] = summary
                    df.loc[q_idx, "risk_or_distress"] = risk
                    df.loc[q_idx, "answer_coverage"] = coverage
            
            # Apply to ALL answer rows in this group
            for a_idx in pair["answer_indices"]:
                if a_idx < len(df):
                    df.loc[a_idx, "key_findings"] = summary
                    df.loc[a_idx, "risk_or_distress"] = risk
                    df.loc[a_idx, "answer_coverage"] = coverage

        # Add validity flag
        df["valid_summary"] = df["key_findings"].apply(
            lambda x: len(str(x).split()) >= 3 and 
                     str(x) not in ["", "No clear insight extracted", "Processing failed"] and
                     not str(x).lower().startswith(('analysis', 'the ', 'this '))
        )

        # Summary statistics
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total Q&A pairs processed: {len(qa_pairs)}")
        successful_analyses = len([o for o in outputs_all if o != 'Processing failed'])
        print(f"Successful analyses: {successful_analyses}")
        print(f"Pairs with valid insights: {valid_insight_pairs}")
        print(f"Risk detected in: {risk_detected_pairs} pairs")
        print(f"Fully answered: {fully_answered_pairs} pairs")

        # YOUR EXACT JSON output creation
        print("\nCreating JSON output...")

        def safe_convert(obj):
            """Convert all data types to JSON-safe types"""
            if pd.isna(obj) or obj is None:
                return None
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return str(obj)

        # Create JSON structure
        json_output = {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "model_used": "microsoft/Phi-4-mini-instruct",
                "total_qa_pairs": len(qa_pairs),
                "successful_analyses": successful_analyses,
                "processing_summary": {
                    "pairs_with_valid_insights": valid_insight_pairs,
                    "risk_detected_pairs": risk_detected_pairs,
                    "fully_answered_pairs": fully_answered_pairs
                }
            },
            "qa_analyses": []
        }

        # Add each Q&A pair analysis
        for i, (pair, llm_output) in enumerate(zip(qa_pairs, outputs_all)):
            qa_analysis = {
                "qa_pair_id": i + 1,
                "grouping_info": {
                    "year": safe_convert(pair["year"]),
                    "quarter": safe_convert(pair["quarter"]),
                    "question_no": safe_convert(pair["question_no"])
                },
                "question_text": str(pair["questions"]),
                "answer_text": str(pair["answer"]),
                "raw_llm_output": str(llm_output),
                "extracted_results": {
                    "key_findings": str(self.extract_summary(llm_output) if llm_output != "Processing failed" else ""),
                    "risk_or_distress": str(self.extract_risk(llm_output) if llm_output != "Processing failed" else ""),
                    "answer_coverage": str(self.extract_coverage(llm_output) if llm_output != "Processing failed" else "")
                },
                "processing_status": "success" if llm_output != "Processing failed" else "failed",
                "question_indices": [safe_convert(idx) for idx in pair["question_indices"]],
                "answer_indices": [safe_convert(idx) for idx in pair["answer_indices"]]
            }
            json_output["qa_analyses"].append(qa_analysis)

        # Return results for Streamlit
        return {
            "metadata": json_output["metadata"],
            "qa_analyses": json_output["qa_analyses"],
            "risk_detected_pairs": risk_detected_pairs,
            "fully_answered_pairs": fully_answered_pairs,
            "valid_insight_pairs": valid_insight_pairs
        }

    def answer_question(self, question: str, df: pd.DataFrame) -> str:
        """
        Answer a question about the banking data
        """
        # Create context from data
        context_info = self._create_context(df)
        
        prompt = f"""You are a banking AI assistant analyzing HSBC financial data. Answer the following question based on the provided context.

Context Information:
{context_info}

Question: {question}

Answer:"""
        
        try:
            with torch.no_grad():
                response = self.pipe(prompt, **self.generation_args)
                if isinstance(response, list) and len(response) > 0:
                    return response[0]['generated_text'].strip()
                else:
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_context(self, df: pd.DataFrame) -> str:
        """Create context information from DataFrame"""
        context = []
        
        # Basic statistics
        context.append(f"Total records: {len(df)}")
        context.append(f"Unique speakers: {df['Speaker name'].nunique()}")
        context.append(f"Q&A records: {len(df[df['Question No'].notna()])}")
        
        # Year/Quarter info
        if 'Year' in df.columns:
            years = df['Year'].dropna().unique()
            context.append(f"Years covered: {', '.join(map(str, sorted(years)))}")
        
        if 'Quarter' in df.columns:
            quarters = df['Quarter'].dropna().unique()
            context.append(f"Quarters: {', '.join(map(str, quarters))}")
        
        # Top speakers
        top_speakers = df['Speaker name'].value_counts().head(5)
        context.append(f"Top speakers: {', '.join(top_speakers.index.tolist())}")
        
        return "\n".join(context)