import re
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import unicodedata

def clean_text_for_finbert(
    texts: List[str],
    preserve_financial_entities: bool = True,
    normalize_numbers: bool = True,
    preserve_financial_punctuation: bool = True,
    remove_html: bool = True,
    handle_financial_abbreviations: bool = True,
    max_length: int = 512,
    min_length: int = 10
) -> List[str]:
    """
    Clean financial text data for FinBERT processing.
    
    Args:
        texts: List of financial text documents
        preserve_financial_entities: Keep financial entities like tickers, currencies
        normalize_numbers: Standardize financial numbers and percentages
        preserve_financial_punctuation: Keep punctuation important for financial context
        remove_html: Remove HTML tags and entities
        handle_financial_abbreviations: Expand common financial abbreviations
        max_length: Maximum text length (FinBERT limit is 512 tokens)
        min_length: Minimum text length to keep
    
    Returns:
        List of cleaned financial texts
    """
    
    # Financial abbreviations dictionary
    financial_abbrevs = {
        r'\bQ1\b': 'first quarter',
        r'\bQ2\b': 'second quarter', 
        r'\bQ3\b': 'third quarter',
        r'\bQ4\b': 'fourth quarter',
        r'\bYoY\b': 'year over year',
        r'\bQoQ\b': 'quarter over quarter',
        r'\bEBITDA\b': 'earnings before interest taxes depreciation amortization',
        r'\bROE\b': 'return on equity',
        r'\bROA\b': 'return on assets',
        r'\bP/E\b': 'price to earnings',
        r'\bEPS\b': 'earnings per share',
        r'\bIPO\b': 'initial public offering',
        r'\bM&A\b': 'mergers and acquisitions',
        r'\bCEO\b': 'chief executive officer',
        r'\bCFO\b': 'chief financial officer',
        r'\bFY\b': 'fiscal year',
        r'\bB\b(?=\s|$)': 'billion',
        r'\bM\b(?=\s|$)': 'million',
        r'\bK\b(?=\s|$)': 'thousand'
    }
    
    cleaned_texts = []
    
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
            
        clean_text = str(text).strip()
        
        # Remove HTML tags and entities
        if remove_html:
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean_text)
        
        # Handle financial abbreviations
        if handle_financial_abbreviations:
            for abbrev, expansion in financial_abbrevs.items():
                clean_text = re.sub(abbrev, expansion, clean_text, flags=re.IGNORECASE)
        
        # Handle financial entities preservation
        if preserve_financial_entities:
            # Protect stock tickers during cleaning
            ticker_pattern = r'\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b'
            tickers = re.findall(ticker_pattern, clean_text)
            ticker_placeholders = {}
            
            for i, ticker in enumerate(tickers):
                placeholder = f"__TICKER_{i}__"
                ticker_placeholders[placeholder] = ticker
                clean_text = clean_text.replace(ticker, placeholder, 1)

            # Protect currency symbols and amounts
            currency_pattern = r'[\$€£¥]\d+(?:,\d{3})*(?:\.\d{2})?[BMK]?'
            currencies = re.findall(currency_pattern, clean_text)
            currency_placeholders = {}
            
            for i, currency in enumerate(currencies):
                placeholder = f"__CURRENCY_{i}__"
                currency_placeholders[placeholder] = currency
                clean_text = clean_text.replace(currency, placeholder, 1)
            
            # Protect percentage values
            percentage_pattern = r'\d+(?:\.\d+)?%'
            percentages = re.findall(percentage_pattern, clean_text)
            percentage_placeholders = {}
            
            for i, percentage in enumerate(percentages):
                placeholder = f"__PERCENTAGE_{i}__"
                percentage_placeholders[placeholder] = percentage
                clean_text = clean_text.replace(percentage, placeholder, 1)
        # Normalize financial numbers while preserving meaning
        if normalize_numbers:
            # Handle percentages (keep the % symbol)
            clean_text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', clean_text)
            
            # Handle currency amounts (preserve currency symbols)
            clean_text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'dollar \1', clean_text)
            clean_text = re.sub(r'€(\d+(?:,\d{3})*(?:\.\d{2})?)', r'euro \1', clean_text)
            clean_text = re.sub(r'£(\d+(?:,\d{3})*(?:\.\d{2})?)', r'pound \1', clean_text)
            
            # Normalize large numbers with commas
            clean_text = re.sub(r'(\d+),(\d{3})', r'\1\2', clean_text)
        
        # Preserve important financial punctuation
        if not preserve_financial_punctuation:
            # Only remove if explicitly requested - usually we want to keep financial punctuation
            clean_text = re.sub(r'[^\w\s$€£%.,()-]', ' ', clean_text)
        
        # Clean up whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'\n+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        # Filter by length
        if min_length <= len(clean_text) <= max_length * 4:  # Rough character to token ratio
            cleaned_texts.append(clean_text)
    
    return cleaned_texts

def preprocess_financial_data(
    df: pd.DataFrame,
    text_column: str,
    date_column: Optional[str] = None,
    company_column: Optional[str] = None,
    sentiment_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocess financial DataFrame for FinBERT analysis.
    
    Args:
        df: DataFrame containing financial text data
        text_column: Name of column containing text to analyze
        date_column: Name of date column (for temporal analysis)
        company_column: Name of company/ticker column
        sentiment_column: Name of existing sentiment column (if any)
    
    Returns:
        Preprocessed DataFrame
    """
    
    df_clean = df.copy()
    
    # Clean text column
    df_clean[text_column] = clean_text_for_finbert(df_clean[text_column].fillna('').tolist())
    
    # Handle dates
    if date_column and date_column in df_clean.columns:
        df_clean[date_column] = pd.to_datetime(df_clean[date_column], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_column])
    
    # Standardize company names/tickers
    if company_column and company_column in df_clean.columns:
        df_clean[company_column] = df_clean[company_column].str.upper().str.strip()
    
    # Remove rows with empty text after cleaning
    df_clean = df_clean[df_clean[text_column].str.len() > 0]
    
    return df_clean.reset_index(drop=True)

def extract_financial_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract common financial entities from text.
    Useful for understanding what financial information is present.
    """
    
    entities = {
        'tickers': [],
        'currencies': [],
        'percentages': [],
        'monetary_amounts': [],
        'dates': []
    }
    
    # Stock tickers (basic pattern)
    tickers = re.findall(r'\b[A-Z]{2,5}\b(?:\.[A-Z]{1,2})?', text)
    entities['tickers'] = list(set(tickers))
    
    # Currency amounts
    currencies = re.findall(r'[\$€£¥]\d+(?:,\d{3})*(?:\.\d{2})?[BMK]?', text)
    entities['currencies'] = currencies
    
    # Percentages
    percentages = re.findall(r'\d+(?:\.\d+)?%', text)
    entities['percentages'] = percentages
    
    # Monetary amounts in words
    monetary = re.findall(r'\d+(?:\.\d+)?\s*(?:billion|million|thousand)', text, re.IGNORECASE)
    entities['monetary_amounts'] = monetary
    
    # Dates (basic patterns)
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b', text)
    entities['dates'] = dates
    
    return entities

def prepare_for_finbert_sentiment(
    texts: List[str],
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, any]]:
    """
    Prepare texts for FinBERT sentiment analysis by chunking if necessary.
    
    Args:
        texts: List of financial texts
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with text chunks and metadata
    """
    
    prepared_data = []
    
    for idx, text in enumerate(texts):
        if len(text) <= chunk_size:
            prepared_data.append({
                'original_index': idx,
                'chunk_index': 0,
                'text': text,
                'is_chunked': False
            })
        else:
            # Split into overlapping chunks
            chunks = []
            start = 0
            chunk_idx = 0
            
            while start < len(text):
                end = start + chunk_size
                
                # Try to break at sentence boundary
                if end < len(text):
                    last_period = text.rfind('.', start, end)
                    last_exclamation = text.rfind('!', start, end)
                    last_question = text.rfind('?', start, end)
                    
                    sentence_end = max(last_period, last_exclamation, last_question)
                    if sentence_end > start + chunk_size // 2:
                        end = sentence_end + 1
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    prepared_data.append({
                        'original_index': idx,
                        'chunk_index': chunk_idx,
                        'text': chunk_text,
                        'is_chunked': True
                    })
                    chunk_idx += 1
                
                start = end - overlap
                if start >= len(text):
                    break
    
    return prepared_data

# Example usage with popular FinBERT libraries
def example_finbert_usage():
    """
    Example of how to use cleaned data with FinBERT models.
    """
    
    # Sample financial texts
    financial_texts = [
        "AAPL reported Q3 earnings of $1.30 per share, beating estimates by 15%. Revenue increased 8.5% YoY to $81.4B.",
        "The Federal Reserve announced a 0.25% interest rate hike, citing persistent inflation concerns.",
        "Tesla's stock price dropped 12% after CEO comments about production challenges in Q4.",
        "JPMorgan's ROE improved to 15.2% this quarter, with strong performance in investment banking division."
    ]
    
    # Clean the texts
    cleaned_texts = clean_text_for_finbert(financial_texts)
    
    print("Cleaned texts for FinBERT:")
    for i, text in enumerate(cleaned_texts):
        print(f"{i+1}: {text}")
        
        # Extract financial entities
        entities = extract_financial_entities(text)
        print(f"   Entities: {entities}")
        print()

if __name__ == "__main__":
    example_finbert_usage()