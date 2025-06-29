"""
STREAMLINED Phi-4 Method 2 Agent - Same prompts/tokens, faster processing
Fixes processing bottlenecks while keeping original AI functionality intact
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import gc
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import streamlit as st

class Phi4Method2Agent:
    """
    STREAMLINED Phi-4 Agent for Method 2 - Same prompts, faster processing
    """
    
    def __init__(self):
        """Initialize the Phi-4 model for Method 2 analysis"""
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_loaded = False
        
        # ORIGINAL generation args - UNCHANGED
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        self.batch_size = 8 if torch.cuda.is_available() else 2
        
        # ORIGINAL standard message - UNCHANGED
        self.standard_message = [
            {
                'role': 'system',
                'content': (
                    "You are a financial analyst. For each Q&A exchange, extract only the financial metric(s) discussed, "
                    "provide numerical value if mentioned, any future trajectory (e.g., increase, decline, flat), and whether "
                    "the presenters' responses answered the questions or avoided them. \n\n"
                    "⚠️ Format your answer strictly as a flat list of strings like this:\n"
                    "['Metric1', 'LEVEL or NO LEVEL', 'TRAJECTORY or NO TRAJECTORY', 'ANSWERED or AVOIDED', "
                    "'Metric2', ...].\n"
                    "❌ Do NOT return dictionaries, JSON objects, or multi-line strings.\n"
                    "✅ Return a *single flat list* of values in the correct order."
                )
            },
            {
                'role': 'user',
                'content': (
                    "From the following query object [{'type': 'query', 'person_type': 'participant', 'name': 'Russ Hunter', "
                    "'text': 'My first question relate to FTSE 100, starting at the level of 1500 points, do you think it will "
                    "grow in medium term? And my second question refers to both ROE and ROA? Do you think they will increase "
                    "or decrease?'}] and the following answer objects [{'type': 'answer', 'person_type': 'presenter',"
                    "'name': 'Olivia Hunter', 'job': 'CFO', 'text': 'The FTSE 100 has some really complex visualisations. "
                    "Related to your second question we expect ROE to increase at 5% and ROA to decrease.'}], "
                    "extract each financial metric mentioned, the level if mentioned (or NO LEVEL), future trajectory "
                    "(or NO TRAJECTORY), and whether the question was ANSWERED or AVOIDED. Return result as a flat list."
                )
            },
            {
                'role': 'assistant',
                'content': (
                    "['FTSE 100', '1500', 'GROW', 'AVOIDED', 'ROE', 'NO LEVEL', 'INCREASE', 'ANSWERED', "
                    "'ROA', '5%', 'DECREASE', 'ANSWERED']"
                )
            }
        ]
    
    def _load_model_lazy(self):
        """STREAMLINED: Faster model loading with less overhead"""
        if self.model_loaded:
            return True
            
        try:
            # Check session state cache first
            if hasattr(st.session_state, 'phi4_method2_model_cache') and st.session_state.phi4_method2_model_cache is not None:
                cached_model = st.session_state.phi4_method2_model_cache
                self.model = cached_model['model']
                self.tokenizer = cached_model['tokenizer']
                self.pipe = cached_model['pipe']
                self.model_loaded = True
                return True
            
            # STREAMLINED: Simpler loading with minimal UI
            with st.spinner("🤖 Loading Phi-4 Method 2 model..."):
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
                
                if torch.cuda.is_available():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Phi-4-mini-instruct",
                        device_map="cuda",
                        torch_dtype="auto",
                        trust_remote_code=True,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Phi-4-mini-instruct",
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                    )
                
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                
                # Cache in session state
                st.session_state.phi4_method2_model_cache = {
                    'model': self.model,
                    'tokenizer': self.tokenizer,
                    'pipe': self.pipe
                }
                
                self.model_loaded = True
                st.success("✅ Phi-4 Method 2 model loaded and cached!")
                return True
            
        except Exception as e:
            st.error(f"❌ Failed to load Phi-4 Method 2 model: {str(e)}")
            return False

    def _create_message_prompts_fast(self, df: pd.DataFrame, max_pairs: Optional[int] = None) -> pd.DataFrame:
        """STREAMLINED: Faster message creation with less overhead"""
        
        # STREAMLINED: Minimal filtering and copying
        df_messages = df[df['type'].notnull()].copy()
        df_messages = df_messages.sort_index()
        
        if len(df_messages) == 0:
            return df_messages
        
        # STREAMLINED: Pre-allocate for better performance
        df_messages['message_final'] = None
        
        # STREAMLINED: Simplified processing with minimal logging
        processed_pairs = 0
        i = 0
        
        while i < len(df_messages):
            if max_pairs is not None and processed_pairs >= max_pairs:
                break
                
            current_row = df_messages.iloc[i]
            
            if current_row['type'] != 'query':
                i += 1
                continue
            
            # STREAMLINED: Faster object creation
            query_obj = {
                "type": "query",
                "person_type": current_row.get('person_type', 'participant'),
                "name": current_row.get('name', 'Unknown'),
                "text": current_row.get('text', '')
            }
            
            # STREAMLINED: Faster answer collection
            answer_objs = []
            j = i + 1
            
            while j < len(df_messages):
                next_row = df_messages.iloc[j]
                
                if next_row['type'] == 'query':
                    break
                elif next_row['type'] == 'answer':
                    answer_objs.append({
                        "type": "answer",
                        "person_type": next_row.get('person_type', 'presenter'),
                        "name": next_row.get('name', 'Unknown'),
                        "job": next_row.get('job', 'Unknown'),
                        "text": next_row.get('text', '')
                    })
                j += 1
            
            # Only process if we have answers
            if len(answer_objs) > 0:
                # ORIGINAL message creation - UNCHANGED
                message_user = {
                    "role": "user",
                    "content": (
                        f"From the following query object {repr([query_obj])} and the following answer objects "
                        f"{repr(answer_objs)}, extract each financial metric mentioned, the level if mentioned (or NO LEVEL), "
                        f"future trajectory (or NO TRAJECTORY), and whether the question was ANSWERED or AVOIDED. "
                        f"Return the result as a single flat list in the format: "
                        f"['Metric', 'LEVEL or NO LEVEL', 'TRAJECTORY or NO TRAJECTORY', 'ANSWERED or AVOIDED', ...]."
                    )
                }
                
                # ORIGINAL message final - UNCHANGED
                message_final = self.standard_message + [message_user]
                
                # Save to DataFrame
                query_idx = df_messages.index[i]
                df_messages.at[query_idx, 'message_final'] = message_final
                
                processed_pairs += 1
            
            i = j if j < len(df_messages) else len(df_messages)
        
        # STREAMLINED: Minimal UI feedback
        return df_messages

    def analyze_multibank_data(self, df: pd.DataFrame, max_pairs: Optional[int] = None) -> Dict[str, Any]:
        """STREAMLINED: Much faster analysis with reduced UI overhead"""
        
        if not self._load_model_lazy():
            return self._create_empty_results()
        
        # STREAMLINED: Minimal status messages
        total_records = len(df)
        
        # STREAMLINED: Fast message creation without expander overhead
        df_messages = self._create_message_prompts_fast(df, max_pairs=max_pairs)
        
        if len(df_messages) == 0 or df_messages['message_final'].isna().all():
            st.warning("No valid message prompts created")
            return self._create_empty_results()
        
        # STREAMLINED: Quick filtering
        valid_messages = df_messages[df_messages['message_final'].notna()]
        
        if len(valid_messages) == 0:
            st.warning("No valid message prompts found")
            return self._create_empty_results()
        
        total_pairs = len(valid_messages)
        
        # STREAMLINED: Process with minimal UI overhead
        results = []
        
        # STREAMLINED: Only show progress for larger datasets
        if total_pairs > 3:
            progress_bar = st.progress(0)
            show_progress = True
        else:
            show_progress = False
        
        # STREAMLINED: Direct processing without excessive logging
        for i, (idx, row) in enumerate(valid_messages.iterrows()):
            try:
                message_final = row['message_final']
                
                if message_final is not None and isinstance(message_final, list):
                    # ORIGINAL AI processing - UNCHANGED
                    result = self.pipe(message_final, **self.generation_args)
                    generated_text = result[0]['generated_text']
                else:
                    generated_text = "ERROR: Invalid message format"
                
                # STREAMLINED: Faster metrics parsing
                metrics_found = self._count_metrics_from_output(generated_text)
                answered = 1 if 'ANSWERED' in generated_text else 0
                avoided = 1 if 'AVOIDED' in generated_text else 0
                
                # STREAMLINED: Faster text extraction
                query_text = row.get('text', 'No question text')
                
                # STREAMLINED: Simplified answer extraction
                answer_texts = []
                if message_final and len(message_final) > 3:
                    user_content = message_final[-1]['content']
                    answer_match = re.search(r'answer objects (\[.*?\])', user_content)
                    if answer_match:
                        try:
                            import ast
                            answer_objs = ast.literal_eval(answer_match.group(1))
                            answer_texts = [ans.get('text', '') for ans in answer_objs if isinstance(ans, dict)]
                        except:
                            answer_texts = ['Answer text parsing failed']
                
                results.append({
                    "qa_pair_id": i + 1,
                    "bank": row.get('bank', 'Unknown'),
                    "year": row.get('year', 'Unknown'),
                    "quarter": row.get('quarter', 'Unknown'),
                    "query_text": query_text,
                    "answer_texts": answer_texts,
                    "raw_llm_output": generated_text,
                    "metrics_extracted": metrics_found,
                    "question_answered": answered > 0,
                    "question_avoided": avoided > 0,
                    "processing_status": "success"
                })
                
                # STREAMLINED: Update progress less frequently
                if show_progress:
                    progress_bar.progress((i + 1) / total_pairs)
                
            except Exception as e:
                # STREAMLINED: Simple error handling
                results.append({
                    "qa_pair_id": i + 1,
                    "bank": row.get('bank', 'Unknown'),
                    "year": row.get('year', 'Unknown'), 
                    "quarter": row.get('quarter', 'Unknown'),
                    "processing_status": "failed",
                    "metrics_extracted": 0,
                    "question_answered": False,
                    "question_avoided": False,
                    "raw_llm_output": "Processing failed",
                    "query_text": "Processing failed",
                    "answer_texts": []
                })
        
        if show_progress:
            progress_bar.empty()
        
        # STREAMLINED: Quick summary
        successful_analyses = len([r for r in results if r['processing_status'] == 'success'])
        total_metrics = sum(r['metrics_extracted'] for r in results)
        answered_questions = sum(1 for r in results if r['question_answered'])
        avoided_questions = sum(1 for r in results if r['question_avoided'])
        
        # STREAMLINED: Single success message
        st.success(f"✅ Processed {total_pairs} Q&A pairs | Metrics: {total_metrics} | Answered: {answered_questions} | Avoided: {avoided_questions}")
        
        return {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "model_used": "microsoft/Phi-4-mini-instruct (Method 2)",
                "total_qa_pairs": total_pairs,
                "analyzed_qa_pairs": total_pairs,
                "successful_analyses": successful_analyses,
                "max_pairs_limit": max_pairs,
                "processing_summary": {
                    "extracted_metrics": total_metrics,
                    "answered_questions": answered_questions,
                    "avoided_questions": avoided_questions
                }
            },
            "qa_analyses": results,
            "extracted_metrics": total_metrics,
            "answered_questions": answered_questions,
            "avoided_questions": avoided_questions
        }
    
    def _count_metrics_from_output(self, output: str) -> int:
        """ORIGINAL metrics counting - UNCHANGED"""
        if not output or output == "Processing failed":
            return 0
        
        try:
            # Look for list-like patterns in the output
            if '[' in output and ']' in output:
                # Extract the list part
                start = output.find('[')
                end = output.rfind(']') + 1
                list_str = output[start:end]
                
                # Try to evaluate the list
                import ast
                try:
                    metric_list = ast.literal_eval(list_str)
                    if isinstance(metric_list, list):
                        # Count groups of 4 (metric, level, trajectory, status)
                        return len(metric_list) // 4
                except:
                    pass
            
            # Fallback: count occurrences of common financial metrics
            metrics_patterns = [
                r'\bROE\b', r'\bROA\b', r'\bNIM\b', r'\bCET1\b', r'\bTier\s*1\b',
                r'\brevenue\b', r'\bprofit\b', r'\bearnings\b', r'\bcapital\b',
                r'\bdividend\b', r'\bFTSE\b', r'\bS&P\b', r'\bNASDAQ\b'
            ]
            
            count = 0
            for pattern in metrics_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                count += len(matches)
            
            return count
            
        except Exception:
            return 0
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results structure"""
        return {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "model_used": "microsoft/Phi-4-mini-instruct (Method 2)",
                "total_qa_pairs": 0,
                "successful_analyses": 0
            },
            "qa_analyses": [],
            "extracted_metrics": 0,
            "answered_questions": 0,
            "avoided_questions": 0
        }

    def answer_question(self, question: str, df: pd.DataFrame) -> str:
        """Answer a question about the multi-bank data"""
        if not self._load_model_lazy():
            return "Model failed to load. Please try again."
        
        context_info = self._create_context(df)
        
        prompt = f"""Based on this multi-bank financial data, answer the question:

Context: {context_info[:500]}...

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
        """Create context information from multi-bank DataFrame"""
        context = []
        
        # Basic statistics
        context.append(f"Dataset contains {len(df):,} records")
        context.append(f"Banks: {', '.join(df['bank'].unique()) if 'bank' in df.columns else 'Unknown'}")
        context.append(f"Years: {', '.join(map(str, sorted(df['year'].unique()))) if 'year' in df.columns else 'Unknown'}")
        
        # Q&A specific info
        if 'type' in df.columns:
            query_count = len(df[df['type'] == 'query'])
            answer_count = len(df[df['type'] == 'answer'])
            context.append(f"Q&A pairs: {query_count} questions, {answer_count} answers")
        
        return ". ".join(context)