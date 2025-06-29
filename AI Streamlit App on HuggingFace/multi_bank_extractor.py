"""
Multi-Bank Data Extractor for Method 2 Processing
STREAMS DIRECTLY from Google Drive into memory without downloading files
"""

import pandas as pd
import numpy as np
import json
import re
import unicodedata
from typing import Dict, List, Any, Optional
import requests
import streamlit as st

class MultiBankDataExtractor:
    """
    Multi-bank data extractor - STREAMS DIRECTLY from Google Drive, NO FILE SAVING
    """
    
    def __init__(self):
        """Initialize with bank mappings and position databases"""
        self.bank_names = {
            'barc': 'Barclays',
            'cs': 'Credit Suisse/UBS', 
            'db': 'Deutsche Bank',
            'hsbc': 'HSBC',
            'jpm': 'JP Morgan',
            'san': 'Santander',
            'uc': 'UniCredit'
        }
        
        # File IDs for streaming real data from Google Drive
        self.file_ids = [
            '1XhLXQV6sQxiHsjhAmdHAp9cKrxY_njeU', # Barclays
            '1fhmMml1irnFwwOWtdE5N_yjn8ZUDjCIX', # Credit Suisse
            '1pcVpQ3SOHFtOuZA34W0akKxdkjh0-WCd', # Deutsche Bank
            '1EiSRaSu0fFj_yYtnBA8GuaoEo26WfAPS', # HSBC
            '1JdxdmL4ND-yVH6HvWcm8MPivBEQghi_7', # JPM
            '1kQj4HZxq0C1BqOL6gaoago7S22zyb8Nh', # Santander
            '10bjoeBGZ9tX9eoaNwOXjEX2_L_lWLxGB'  # Unicredit
        ]
        
        self.file_names = ['barc', 'cs', 'db', 'hsbc', 'jpm', 'san', 'uc']
        
        # Position databases for executives
        self.pos_db = {
            "Silke Szypa": "Deputy Head of Investor Relations",
            "Christian Sewing": "Chief Executive Officer", 
            "James von Moltke": "Chief Financial Officer",
            "Ioana Patriniche": "Head of Investor Relations"
        }
        
        self.pos_hsbc = {
            "Georges Elhedery": "Group Chief Executive Officer",
            "Pam Kaur": "Group Chief Financial Officer",
            "Colin Bell": "CEO, HSBC Bank plc & Europe",
            "Greg Guyett": "CEO, Global Banking & Markets",
            "Nuno Matos": "CEO, Wealth & Personal Banking"
        }
        
        self.pos_jpm = {
            "Jamie Dimon": "Chairman & CEO",
            "Jeremy Barnum": "Chief Financial Officer",
            "Marianne Lake": "Co‑CEO, Consumer & Community Banking",
            "Jennifer Piepszak": "Chief Financial Officer, Corporate & Investment Bank"
        }
        
        self.pos_cs = {
            "Ulrich Korner": "Group Chief Executive Officer",
            "Ulrich Koerner": "Group Chief Executive Officer", 
            "Dixit Joshi": "Group Chief Financial Officer",
            "Thomas Gottstein": "Former Chief Executive Officer",
            "David Mathers": "Former Chief Financial Officer",
            "Axel Lehmann": "Chairman of Credit Suisse",
            "Sergio P. Ermotti": "Group Chief Executive Officer",
            "Todd Tuckner": "Group Chief Financial Officer",
            "Colm Kelleher": "Chairman of the Board",
            "George Athanasopoulos": "Co‑President Investment Bank",
            "Aleksandar Ivanovic": "President Asset Management",
            "Iqbal Khan": "Co‑President Global Wealth Management & President Asia Pacific",
            "Robert Karofsky": "Co‑President Investment Bank & President UBS Americas",
            "Kirt Gardner": "Group Chief Financial Officer (prior to Tuckner)",
            "Kinner Lakhani": "Chief Financial Officer, Global Wealth Management"
        }
        
        self.pos_uc = {
            "Andrea Orcel": "Chief Executive Officer",
            "Stefano Porro": "Chief Financial Officer", 
            "Stefano Porro ": "Chief Financial Officer",
            "Magda Palczynska": "Head of Investor Relations"
        }
        
        self.pos_san = {
            "Hector Grisi": "Chief Executive Officer",
            "Hector Grisi Checa": "Chief Executive Officer",
            "Jose Garcia": "Chief Financial Officer",
            "Mario Leao": "Chief Executive Officer (Brasil)",
            "Gustavo Alejo": "Chief Financial Officer (Brasil)"
        }
        
        self.pos_barc = {
            "C.S. Venkatakrishnan": "Group Chief Executive",
            "Anna Cross": "Group Finance Director", 
            "Mark Mason": "Group Chief Financial Officer"
        }
        
        self.pos_all = {
            'db': self.pos_db, 
            'hsbc': self.pos_hsbc, 
            'jpm': self.pos_jpm,
            'cs': self.pos_cs, 
            'uc': self.pos_uc, 
            'san': self.pos_san, 
            'barc': self.pos_barc
        }
        
        # ONLY in-memory cache
        self.df_all_cache = None
        self.filters_cache = None
    
    def clear_cache(self):
        """Clear in-memory cache only"""
        self.df_all_cache = None
        self.filters_cache = None
        st.success("✅ In-memory cache cleared successfully!")
    
    def remove_diacritics(self, input_str: str) -> str:
        """Remove diacritics from string"""
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return ''.join(c for c in nfkd_form if not unicodedata.combining(c) and (c.isalpha() or c.isspace() or c == '.'))
    
    def standardize_name(self, name: str) -> str:
        """Standardize speaker names using regex patterns"""
        name = name.strip()
        name = self.remove_diacritics(name)
        
        # All the standardization patterns
        if re.search(r'(?:C\.?\s?S\.?\s?)?Venkatakrishnan$', name, re.IGNORECASE):
            return 'C.S. Venkatakrishnan'
        elif re.search(r'Anna\s?[-_]?Cross$', name, re.IGNORECASE):
            return 'Anna Cross'
        elif re.search(r'Silke\s?Szypa$', name, re.IGNORECASE):
            return 'Silke Szypa'
        elif re.search(r'Christian\s?Sewing$', name, re.IGNORECASE):
            return 'Christian Sewing'
        elif re.search(r'James\s?von\s?Moltke$', name, re.IGNORECASE):
            return 'James von Moltke'
        elif re.search(r'Ioana\s?Patriniche$', name, re.IGNORECASE):
            return 'Ioana Patriniche'
        elif re.search(r'Georges\s?Elhedery$', name, re.IGNORECASE):
            return 'Georges Elhedery'
        elif re.search(r'Pam\s?Kaur$', name, re.IGNORECASE):
            return 'Pam Kaur'
        elif re.search(r'Colin\s?Bell$', name, re.IGNORECASE):
            return 'Colin Bell'
        elif re.search(r'Greg\s?Guyett$', name, re.IGNORECASE):
            return 'Greg Guyett'
        elif re.search(r'Nuno\s?Matos$', name, re.IGNORECASE):
            return 'Nuno Matos'
        elif re.search(r'Jamie\s?Dimon$', name, re.IGNORECASE):
            return 'Jamie Dimon'
        elif re.search(r'Jeremy\s?Barnum$', name, re.IGNORECASE):
            return 'Jeremy Barnum'
        elif re.search(r'Marianne\s?Lake$', name, re.IGNORECASE):
            return 'Marianne Lake'
        elif re.search(r'Jennifer\s?Piepszak$', name, re.IGNORECASE):
            return 'Jennifer Piepszak'
        elif re.search(r'Ulrich\s?Korner$', name, re.IGNORECASE):
            return 'Ulrich Korner'
        elif re.search(r'Ulrich\s?Koerner$', name, re.IGNORECASE):
            return 'Ulrich Korner'
        elif re.search(r'Dixit\s?Joshi$', name, re.IGNORECASE):
            return 'Dixit Joshi'
        elif re.search(r'Thomas\s?Gottstein$', name, re.IGNORECASE):
            return 'Thomas Gottstein'
        elif re.search(r'David\s?Mathers$', name, re.IGNORECASE):
            return 'David Mathers'
        elif re.search(r'Axel\s?Lehmann$', name, re.IGNORECASE):
            return 'Axel Lehmann'
        elif re.search(r'Andrea\s?Orcel$', name, re.IGNORECASE):
            return 'Andrea Orcel'
        elif re.search(r'Stefano\s?Porro$', name, re.IGNORECASE):
            return 'Stefano Porro'
        elif re.search(r'Magda\s?Palczynska$', name, re.IGNORECASE):
            return 'Magda Palczynska'
        elif re.search(r'Hector\s?Grisi(\s\w+)*$', name, re.IGNORECASE):
            return 'Hector Grisi'
        elif re.search(r'Jose\s?Garcia', name, re.IGNORECASE):
            return 'Jose Garcia'
        elif re.search(r'Mario\s?Leao$', name, re.IGNORECASE):
            return 'Mario Leao'
        elif re.search(r'Gustavo\s?Alejo$', name, re.IGNORECASE):
            return 'Gustavo Alejo'
        elif re.search(r'Mark\s?Mason$', name, re.IGNORECASE):
            return 'Mark Mason'
        elif re.search(r'Sergio\s?P\.?\s?Ermotti$', name, re.IGNORECASE):
            return 'Sergio P. Ermotti'
        elif re.search(r'Todd\s?Tuckner$', name, re.IGNORECASE):
            return 'Todd Tuckner'
        elif re.search(r'(A\s*[-–]\s*)?(Angela|Anna)\s?[-_]?Cross$', name, re.IGNORECASE):
            return 'Anna Cross'
        else:
            return name
    
    def detect_question(self, df: pd.DataFrame) -> pd.Series:
        """Detect questions from analysts that contain '?'"""
        return ((df['text'].astype(str).str.contains(r'\?')) &
                (df['title'].astype(str).str.lower().str.contains('analyst'))).astype(int)
    
    def qa_sect_detect(self, df: pd.DataFrame) -> pd.Series:
        """Detect Q&A sections in transcripts"""
        df = df.copy()
        df['flag_qa_sect'] = 0
        
        qa_start_pattern = (r'\b('
            r'start|begin|ready|open.*(q&a|questions|question[-\s]?and[-\s]?answer)|'
            r'first\s+question.*(comes|is|today)|'
            r'(our|we\s+already\s+have\s+the)\s+first\s+question.*(from|comes|is|today)|'
            r'kick.*off.*questions|'
            r'we(\'ll)?\s+start.*(questions|q&a)|'
            r'let(\'s)?\s+start.*(questions|q&a)|'
            r'we(\'ll)?\s+now\s+(begin|open).*question[-\s]?and[-\s]?answer|'
            r'we(\'ll)?\s+take\s+(the\s+)?(next\s+)?questions?|'
            r'we(\'re)?\s+ready\s+to\s+(take|start)\s+(your\s+)?questions?|'
            r'begin\s+with\s+the\s+first\s+question'
            r')\b')
        
        qa_end_pattern = (r'\b('
            r'no\s+(more|further)\s+questions|'
            r'last\s+question|'
            r'(concludes|ends)\s+(the\s+)?(q&a|question[-\s]?and[-\s]?answer|session|call)|'
            r'end\s+(of\s+)?(q&a|question[-\s]?and[-\s]?answer|session|call)|'
            r'(this\s+)?(concludes|ends)\s+(today(\'s)?\s+)?(call|conference(\s+call)?|session)|'
            r'closing\s+remarks|'
            r'(the\s+)?conference\s+(is\s+)?(now\s+)?over|'
            r'(you\s+may\s+)?disconnect(\s+your\s+telephones|)?|'
            r'thank\s+(you|everyone|everybody|all)\s+(again\s+)?for\s+(joining|participating|being\s+with\s+us)|'
            r'thank\s+you\s+for\s+joining\s+(today(\'s)?\s+)?call|'
            r'a\s+recording\s+of\s+the\s+(presentation|call)\s+will\s+be\s+available|'
            r'(you\s+may\s+)?(now\s+)?disconnect(\s+all)?'
            r')\b')
        
        # Create speaker_role column based on title
        df['speaker_role'] = df['title'].str.contains('analyst', case=False, na=False).map({True: 'analyst', False: 'non-analyst'})
        
        df['speaker_lc'] = df['speaker'].str.lower()
        df['text_lc'] = df['text'].str.lower()
        group_cols = ['year', 'quarter', 'bank']
        
        for _, group_df in df.groupby(group_cols):
            group_idx = group_df.index
            sub_df = df.loc[group_idx]
            
            # Look for Operator-based start
            qa_start_match = sub_df[
                (sub_df['speaker_lc'] == 'operator') &
                sub_df['text_lc'].str.contains(qa_start_pattern, regex=True)
            ]
            
            qa_start_idx = qa_start_match.index.min() if not qa_start_match.empty else None
            
            # Fallback: use first speaker
            if qa_start_idx is None:
                fallback_start_match = sub_df[
                    sub_df['text_lc'].str.contains(qa_start_pattern, regex=True)
                ]
                if fallback_start_match.empty:
                    continue
                qa_start_idx = fallback_start_match.index.min()
            
            # Look for Operator end
            qa_end_match = sub_df[
                (sub_df.index > qa_start_idx) &
                (sub_df['speaker_lc'] == 'operator') &
                sub_df['text_lc'].str.contains(qa_end_pattern, regex=True)
            ]
            
            qa_end_idx = qa_end_match.index.min() if not qa_end_match.empty else None
            
            # Fallback: use any speaker
            if qa_end_idx is None:
                fallback_end_match = sub_df[
                    (sub_df.index > qa_start_idx) &
                    sub_df['text_lc'].str.contains(qa_end_pattern, regex=True)
                ]
                qa_end_idx = fallback_end_match.index.min() if not fallback_end_match.empty else group_idx.max() + 1
            
            # Flag rows in the Q&A range
            df.loc[(df.index >= qa_start_idx) & (df.index < qa_end_idx), 'flag_qa_sect'] = 1
        
        return df['flag_qa_sect']
    
    def adjust_flag_question_for_analysts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust question flags for analysts without '?' but in Q&A section"""
        df = df.copy()
        
        analyst_condition = df['title'].str.strip().str.lower() == 'analyst'
        
        condition = (
            (df['flag_qa_sect'] == 1) &
            analyst_condition &
            (df['flag_question'] == 0) &
            (df['text'].str.split().str.len() >= 16)
        )
        
        df.loc[condition, 'flag_question'] = 1
        return df
    
    def load_real_data(self) -> pd.DataFrame:
        """STREAM DIRECTLY: Read JSON data from Google Drive into memory without downloading files"""
        
        # Return cached data if available
        if self.df_all_cache is not None:
            print("Using cached data from memory")
            return self.df_all_cache
        
        print("🔄 Streaming bank data directly from Google Drive into memory...")
        
        json_data = {}
        
        # Stream files directly from Google Drive without saving to disk
        for file_id, name in zip(self.file_ids, self.file_names):
            try:
                # Create direct download URL for Google Drive
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
                print(f"📥 Streaming {name} from Google Drive...")
                
                # Stream content directly into memory
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Parse JSON directly from response content
                json_data[name] = response.json()
                print(f"✅ Loaded {name} into memory successfully")
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error downloading {name}: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error for {name}: {e}")
                continue
            except Exception as e:
                print(f"❌ Failed to stream {name}: {e}")
                continue
        
        if not json_data:
            print("❌ All Google Drive files failed to stream")
            print("✅ Using enhanced sample data instead")
            return self.create_sample_data()
        
        # Process data using correct implementation logic
        tr_split_all = {}
        
        for bank_key in self.file_names:
            bank_data = json_data.get(bank_key, {})
            
            tr_split_all[bank_key] = {
                quarter: content['transcript_split']
                for quarter, content in bank_data.items()
                if 'transcript_split' in content
            }
        
        # Add position information
        for bank, quarters in tr_split_all.items():
            position_lookup = self.pos_all.get(bank, {})
            
            for quarter, entries in quarters.items():
                for i, entry in enumerate(entries):
                    speaker = entry.get('speaker', '')
                    standard_speaker = self.standardize_name(speaker)
                    
                    if standard_speaker in position_lookup:
                        title = position_lookup[standard_speaker]
                    elif speaker.lower() == 'operator':
                        title = 'Operator'
                    else:
                        title = 'Analyst'
                    
                    reordered_entry = {
                        'speaker': speaker,
                        'title': title,
                        **{k: v for k, v in entry.items() if k not in ['speaker', 'title']}
                    }
                    
                    entries[i] = reordered_entry
        
        # Convert to DataFrame
        records = []
        for bank, quarters in tr_split_all.items():
            for quarter, entries in quarters.items():
                year = quarter[:4]
                q = quarter[4:]
                for entry in entries:
                    records.append({
                        "year": year,
                        "quarter": q,
                        "bank": bank,
                        "speaker": entry.get("speaker"),
                        "title": entry.get("title"),
                        "text": entry.get("text")
                    })
        
        df_all = pd.DataFrame(records)
        df_all = df_all[['bank', 'year', 'quarter', 'speaker', 'title', 'text']]
        
        # STORE ONLY IN MEMORY CACHE
        self.df_all_cache = df_all
        
        print(f"✅ Loaded {len(df_all)} total records in memory (NO FILES SAVED)")
        return df_all
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic sample data for Method 2 testing"""
        print("✅ Using enhanced sample data for Method 2 testing...")
        sample_data = []
        
        banks = ['hsbc', 'jpm', 'db', 'barc']  # Reduced for faster testing
        years = ['2023', '2024'] 
        quarters = ['Q3', 'Q4']  # Reduced for faster testing
        
        # Realistic financial questions and answers
        qa_templates = [
            {
                'question': 'Can you provide details on your ROE performance this quarter? What was the actual ROE percentage and do you expect it to increase or decrease next quarter?',
                'answer': 'Our ROE this quarter was 14.2%, which represents a solid performance. We expect ROE to increase slightly to around 15% next quarter based on our current projections and cost efficiency initiatives.',
                'metrics': ['ROE', '14.2%', 'INCREASE', 'ANSWERED']
            },
            {
                'question': 'What are your current capital ratios, specifically CET1? How do they compare to regulatory requirements and what is the outlook?',
                'answer': 'Our CET1 ratio stands at 13.8%, well above the regulatory minimum of 10.5%. We anticipate maintaining stable capital ratios around current levels through the rest of the year.',
                'metrics': ['CET1', '13.8%', 'STABLE', 'ANSWERED']
            },
            {
                'question': 'Could you comment on your NIM trends and revenue outlook? Are you seeing pressure on margins?',
                'answer': 'NIM has been under some pressure, currently at 2.1% compared to 2.3% last quarter. Revenue for the quarter was $8.2 billion. We expect some continued pressure on margins in the near term.',
                'metrics': ['NIM', '2.1%', 'DECREASE', 'ANSWERED']
            },
            {
                'question': 'What about dividend policy and capital returns? Any plans for buybacks or dividend increases?',
                'answer': 'We remain committed to our dividend policy. Current dividend yield is 4.2%. Regarding buybacks, we prefer not to provide specific guidance at this time.',
                'metrics': ['dividend', '4.2%', 'NO TRAJECTORY', 'AVOIDED']
            },
            {
                'question': 'How is your cost efficiency program progressing? What are the targets for cost reduction?',
                'answer': 'Our cost-to-income ratio improved to 61% this quarter from 63% previously. We are targeting further improvements to reach 58% by year end through our efficiency initiatives.',
                'metrics': ['cost ratio', '61%', 'DECREASE', 'ANSWERED']
            }
        ]
        
        for bank in banks:
            for year in years:
                for quarter in quarters:
                    # Add presentation section
                    sample_data.extend([
                        {
                            'bank': bank,
                            'year': year, 
                            'quarter': quarter,
                            'speaker': 'CEO',
                            'title': 'Chief Executive Officer',
                            'text': f'Welcome to {self.bank_names.get(bank, bank)} {quarter} {year} earnings call. This has been a strong quarter with solid financial performance across all our key metrics including ROE, capital ratios, and revenue growth. We remain well-positioned for continued success.'
                        },
                        {
                            'bank': bank,
                            'year': year,
                            'quarter': quarter, 
                            'speaker': 'Operator',
                            'title': 'Operator',
                            'text': 'Thank you. We will now begin the question and answer session. Our first question comes from Deutsche Bank.'
                        }
                    ])
                    
                    # Add realistic Q&A pairs
                    for i, qa_template in enumerate(qa_templates[:3]):  # Use first 3 for testing
                        sample_data.extend([
                            {
                                'bank': bank,
                                'year': year,
                                'quarter': quarter,
                                'speaker': f'Analyst {i+1}',
                                'title': 'Analyst',
                                'text': qa_template['question']
                            },
                            {
                                'bank': bank,
                                'year': year,
                                'quarter': quarter,
                                'speaker': 'CFO',
                                'title': 'Chief Financial Officer',
                                'text': qa_template['answer']
                            }
                        ])
                    
                    # Add call end
                    sample_data.append({
                        'bank': bank,
                        'year': year,
                        'quarter': quarter,
                        'speaker': 'Operator',
                        'title': 'Operator',
                        'text': 'Thank you. This concludes today\'s call. You may now disconnect.'
                    })
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Created {len(df)} realistic sample records with proper Q&A structure")
        return df
    
    def process_data_multiselect(self, selected_banks: List[str] = None, selected_years: List[str] = None, selected_quarters: List[str] = None) -> pd.DataFrame:
        """Process the multi-bank data with multi-select filtering options"""
        df_all = self.load_real_data()
        
        print(f"Starting with {len(df_all)} total records")
        
        # Apply filters
        if selected_banks:
            df_all = df_all[df_all['bank'].isin(selected_banks)]
            print(f"After bank filter: {len(df_all)} records")
        if selected_years:
            df_all = df_all[df_all['year'].isin(selected_years)]
            print(f"After year filter: {len(df_all)} records")
        if selected_quarters:
            df_all = df_all[df_all['quarter'].isin(selected_quarters)]
            print(f"After quarter filter: {len(df_all)} records")
        
        # Apply Q&A detection logic
        df_all['flag_question'] = self.detect_question(df_all)
        df_all['flag_qa_sect'] = self.qa_sect_detect(df_all)
        df_all = self.adjust_flag_question_for_analysts(df_all)
        df_all['flag_qa_sect'] = self.qa_sect_detect(df_all)
        
        # Create helper columns
        df_all['flag_sum_question_qa'] = df_all['flag_question'] + df_all['flag_qa_sect']
        df_all['flag_analyst'] = (df_all['title'] == 'Analyst').astype(int)
        df_all['flag_operator'] = (df_all['title'] == 'Operator').astype(int)
        df_all['flag_presenter'] = (~df_all['title'].isin(['Operator', 'Analyst'])).astype(int)
        
        # Create "type" column
        def set_type(row):
            word_count = len(str(row['text']).split())
            
            if word_count <= 20:
                return np.nan
            
            if row['flag_operator'] == 1:
                return np.nan
            elif row['flag_sum_question_qa'] == 2:
                return 'query'
            elif (row['flag_sum_question_qa'] == 1 and 
                  row['flag_analyst'] == 0 and 
                  row['flag_presenter'] == 1):
                return 'answer'
            elif (row['flag_sum_question_qa'] == 1 and 
                  row['flag_analyst'] == 1 and 
                  row['flag_presenter'] == 0):
                return np.nan
            else:
                return np.nan
        
        df_all['type'] = df_all.apply(set_type, axis=1)
        
        # Create additional columns
        df_all['person_type'] = np.where(df_all['flag_analyst'] == 1, 'participant', 
                                        np.where(df_all['flag_presenter'] == 1, 'presenter', None))
        df_all['name'] = np.where(df_all['person_type'].notnull(), df_all['speaker'], None)
        df_all['job'] = np.where(df_all['person_type'] == 'presenter', df_all['title'], None)
        
        print(f"Final processed data: {len(df_all)} records")
        
        # Count Q&A pairs
        query_count = len(df_all[df_all['type'] == 'query'])
        answer_count = len(df_all[df_all['type'] == 'answer'])
        print(f"Found {query_count} queries and {answer_count} answers")
        
        return df_all

    def process_data(self, selected_bank: str = None, selected_year: str = None, selected_quarter: str = None) -> pd.DataFrame:
        """Process the multi-bank data with filtering options"""
        selected_banks = [selected_bank] if selected_bank else None
        selected_years = [selected_year] if selected_year else None
        selected_quarters = [selected_quarter] if selected_quarter else None
        
        return self.process_data_multiselect(selected_banks, selected_years, selected_quarters)
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter options from real data with caching"""
        if self.filters_cache is not None:
            return self.filters_cache
        
        df_sample = self.load_real_data()
        
        self.filters_cache = {
            'banks': sorted(list(df_sample['bank'].unique())),
            'years': sorted(list(df_sample['year'].unique())),
            'quarters': sorted(list(df_sample['quarter'].unique()))
        }
        
        return self.filters_cache
    
    def get_filtered_options(self, selected_banks: List[str] = None, selected_years: List[str] = None) -> Dict[str, List[str]]:
        """Get filtered year and quarter options based on selected banks and years"""
        df_sample = self.load_real_data()
        
        all_banks = sorted(list(df_sample['bank'].unique()))
        
        if selected_banks:
            filtered_df = df_sample[df_sample['bank'].isin(selected_banks)]
            available_years = sorted(list(filtered_df['year'].unique()))
            
            if selected_years:
                year_filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
                available_quarters = sorted(list(year_filtered_df['quarter'].unique()))
            else:
                available_quarters = sorted(list(filtered_df['quarter'].unique()))
        else:
            available_years = sorted(list(df_sample['year'].unique()))
            if selected_years:
                year_filtered_df = df_sample[df_sample['year'].isin(selected_years)]
                available_quarters = sorted(list(year_filtered_df['quarter'].unique()))
            else:
                available_quarters = sorted(list(df_sample['quarter'].unique()))
        
        return {
            'banks': all_banks,
            'years': available_years,
            'quarters': available_quarters
        }
    
    def get_bank_year_quarter_combinations(self) -> Dict[str, Dict[str, List[str]]]:
        """Get all valid bank-year-quarter combinations for advanced filtering"""
        df_sample = self.load_real_data()
        
        combinations = {}
        for bank in df_sample['bank'].unique():
            bank_data = df_sample[df_sample['bank'] == bank]
            bank_combinations = {}
            
            for year in sorted(bank_data['year'].unique()):
                year_data = bank_data[bank_data['year'] == year]
                quarters = sorted(year_data['quarter'].unique())
                bank_combinations[str(year)] = quarters
            
            combinations[bank] = bank_combinations
        
        return combinations
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the processed data"""
        stats = {
            'total_records': len(df),
            'unique_speakers': df['speaker'].nunique() if 'speaker' in df.columns else 0,
            'qa_records': len(df[df['type'] == 'query']) if 'type' in df.columns else 0,
            'unique_banks': df['bank'].nunique() if 'bank' in df.columns else 0,
            'banks_covered': df['bank'].unique().tolist() if 'bank' in df.columns else [],
            'years_covered': sorted(df['year'].unique().tolist()) if 'year' in df.columns else [],
            'quarters_covered': sorted(df['quarter'].unique().tolist()) if 'quarter' in df.columns else []
        }
        
        return stats