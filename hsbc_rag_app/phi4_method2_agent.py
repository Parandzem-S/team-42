"""
Fixed Phi-4 Method 2 Agent for Multi-Bank Financial Analysis
Based on the correct implementation from improved_version_of_the_correct_copy_of_untitled6_v4.py
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
    Fixed Phi-4 Agent for Method 2 - based on correct implementation
    """
    
    def __init__(self):
        """Initialize the Phi-4 model for Method 2 analysis"""
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.model_loaded = False
        
        # Method 2 specific generation args (from working implementation)
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        self.batch_size = 8 if torch.cuda.is_available() else 2
        
        # Standard message from working implementation
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
        """Lazy load model only when needed with proper session state caching"""
        if self.model_loaded:
            return True
            
        try:
            # Check if we already have a loaded model in session state
            if hasattr(st.session_state, 'phi4_method2_model_cache') and st.session_state.phi4_method2_model_cache is not None:
                cached_model = st.session_state.phi4_method2_model_cache
                self.model = cached_model['model']
                self.tokenizer = cached_model['tokenizer']
                self.pipe = cached_model['pipe']
                self.model_loaded = True
                return True
            
            # Only show loading message if we're actually loading for the first time
            with st.spinner("🤖 Loading Phi-4 Method 2 model... (This only happens once)"):
                progress_bar = st.progress(0)
                
                progress_bar.progress(25)
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", trust_remote_code=True)
                
                progress_bar.progress(50)
                
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
                
                progress_bar.progress(75)
                
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                
                progress_bar.progress(100)
                progress_bar.empty()
                
                # Cache the loaded model in session state to avoid reloading
                st.session_state.phi4_method2_model_cache = {
                    'model': self.model,
                    'tokenizer': self.tokenizer,
                    'pipe': self.pipe
                }
                
                self.model_loaded = True
                st.success("✅ Phi-4 Method 2 model loaded and cached successfully!")
                return True
            
        except Exception as e:
            if 'progress_bar' in locals():
                progress_bar.empty()
            st.error(f"❌ Failed to load Phi-4 Method 2 model: {str(e)}")
            return False

    def _create_message_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create message prompts from the dataframe using the correct Q&A pairing logic"""
        # Filter only rows where type is not empty and sort by index to maintain order
        df_messages = df[df['type'].notnull()].copy()
        df_messages = df_messages.sort_index()
        
        if len(df_messages) == 0:
            st.warning("No valid query/answer pairs found after filtering")
            return df_messages
        
        # Add required columns for message creation
        required_cols = ['type', 'person_type', 'name', 'job', 'text', 'bank', 'year', 'quarter']
        for col in required_cols:
            if col not in df_messages.columns:
                df_messages[col] = None
        
        # Initialize the column
        df_messages['message_final'] = pd.Series(dtype='object')
        
        # Process queries and their immediately following answers
        processed_pairs = 0
        i = 0
        
        while i < len(df_messages):
            current_row = df_messages.iloc[i]
            
            # Skip if not a query
            if current_row['type'] != 'query':
                i += 1
                continue
            
            # Found a query - construct the query object
            query_obj = {
                "type": "query",
                "person_type": current_row.get('person_type', 'participant'),
                "name": current_row.get('name', 'Unknown'),
                "text": current_row.get('text', '')
            }
            
            # Gather all consecutive answers that follow this query
            answer_objs = []
            j = i + 1
            
            # Collect all consecutive "answer" rows until we hit another "query" or end of data
            while j < len(df_messages):
                next_row = df_messages.iloc[j]
                
                if next_row['type'] == 'query':
                    # Hit another query, stop collecting answers
                    break
                elif next_row['type'] == 'answer':
                    # This is an answer to our current query
                    answer_objs.append({
                        "type": "answer",
                        "person_type": next_row.get('person_type', 'presenter'),
                        "name": next_row.get('name', 'Unknown'),
                        "job": next_row.get('job', 'Unknown'),
                        "text": next_row.get('text', '')
                    })
                
                j += 1
            
            # Only process if we have at least one answer
            if len(answer_objs) > 0:
                # Create the user content message using the correct format
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
                
                # Final message for Phi-4
                message_final = self.standard_message + [message_user]
                
                # Save to DataFrame at the query row index
                query_idx = df_messages.index[i]
                df_messages.at[query_idx, 'message_final'] = message_final
                
                processed_pairs += 1
                
                # Debug info
                st.write(f"**Q&A Pair {processed_pairs}** - Bank: {current_row.get('bank', 'Unknown')}")
                st.write(f"Query: {query_obj['text'][:100]}...")
                st.write(f"Answers: {len(answer_objs)} answer(s) found")
                st.write("---")
            
            # Move to the next potential query (which is at position j)
            i = j if j < len(df_messages) else len(df_messages)
        
        st.success(f"✅ Created {processed_pairs} Q&A pair prompts for analysis")
        return df_messages

    def analyze_multibank_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze multi-bank transcript data using the correct Q&A pairing logic
        """
        if not self._load_model_lazy():
            return self._create_empty_results()
        
        st.info("🔍 Analyzing multi-bank data with Phi-4 Method 2...")
        st.info(f"📊 Input data: {len(df)} total records")
        
        # Show breakdown of data types
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            st.info(f"🔍 Data breakdown: {dict(type_counts)}")
        
        # Create message prompts using the correct logic
        with st.expander("🔧 Q&A Pair Creation Process", expanded=False):
            df_messages = self._create_message_prompts(df)
        
        if len(df_messages) == 0 or df_messages['message_final'].isna().all():
            st.warning("No valid message prompts created from the data")
            return self._create_empty_results()
        
        # Filter out rows without message_final
        valid_messages = df_messages[df_messages['message_final'].notna()]
        
        if len(valid_messages) == 0:
            st.warning("No valid message prompts found")
            return self._create_empty_results()
        
        total_pairs = len(valid_messages)
        st.success(f"✅ Ready to process {total_pairs} Q&A pairs with Phi-4")
        
        # Process with progress tracking
        results = []
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(valid_messages.iterrows()):
            try:
                message_final = row['message_final']
                
                if message_final is not None and isinstance(message_final, list):
                    result = self.pipe(message_final, **self.generation_args)
                    generated_text = result[0]['generated_text']
                else:
                    generated_text = "ERROR: Invalid message format"
                
                # Parse the output for metrics count
                metrics_found = self._count_metrics_from_output(generated_text)
                answered = 1 if 'ANSWERED' in generated_text else 0
                avoided = 1 if 'AVOIDED' in generated_text else 0
                
                # Extract the original query and find associated answers
                query_text = row.get('text', 'No question text')
                
                # Find associated answer texts by looking at the message_final content
                answer_texts = []
                if message_final and len(message_final) > 3:  # Has user message
                    user_content = message_final[-1]['content']
                    # Extract answer objects from the user content
                    import re
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
                
                progress_bar.progress((i + 1) / total_pairs)
                
            except Exception as e:
                st.warning(f"Failed to process Q&A pair {i+1}: {e}")
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
        
        progress_bar.empty()
        
        # Calculate summary statistics
        successful_analyses = len([r for r in results if r['processing_status'] == 'success'])
        total_metrics = sum(r['metrics_extracted'] for r in results)
        answered_questions = sum(1 for r in results if r['question_answered'])
        avoided_questions = sum(1 for r in results if r['question_avoided'])
        
        # Show summary
        st.success(f"✅ Processed {total_pairs} Q&A pairs successfully!")
        st.info(f"""📊 **Results Summary:**
        • Total Metrics Extracted: {total_metrics}
        • Questions Answered: {answered_questions}
        • Questions Avoided: {avoided_questions}
        • Success Rate: {successful_analyses}/{total_pairs} pairs""")
        
        return {
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "model_used": "microsoft/Phi-4-mini-instruct (Method 2)",
                "total_qa_pairs": total_pairs,
                "analyzed_qa_pairs": total_pairs,
                "successful_analyses": successful_analyses,
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
        """Count financial metrics from the LLM output using the correct format"""
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