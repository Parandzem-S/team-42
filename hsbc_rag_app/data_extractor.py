"""
Simple Data Extractor for HSBC Banking Agent
Minimal additional processing - your core code remains unchanged
"""

import pandas as pd
import re
from typing import List, Dict, Any

class DataExtractor:
    """
    Simple data extractor that works with your existing DataFrame structure
    """
    
    def __init__(self):
        """Initialize with basic patterns"""
        self.amount_pattern = re.compile(r'[\$£€¥]\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*(?:million|billion|k|m|b)', re.IGNORECASE)
        self.percentage_pattern = re.compile(r'\b\d+\.?\d*\s*%', re.IGNORECASE)
    
    def extract_financial_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract basic financial data from text"""
        if pd.isna(text):
            return {'amounts': [], 'percentages': []}
            
        amounts = self.amount_pattern.findall(str(text))
        percentages = self.percentage_pattern.findall(str(text))
        
        return {
            'amounts': amounts,
            'percentages': percentages
        }
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the DataFrame (matches your data structure)"""
        stats = {
            'total_records': len(df),
            'unique_speakers': df['Speaker name'].nunique() if 'Speaker name' in df.columns else 0,
            'qa_records': len(df[df['Question No'].notna()]) if 'Question No' in df.columns else 0,
            'unique_files': df['File'].nunique() if 'File' in df.columns else 0
        }
        
        # Year/Quarter info if available
        if 'Year' in df.columns:
            years = df['Year'].dropna().unique()
            stats['years_covered'] = sorted(years.tolist()) if len(years) > 0 else []
        
        if 'Quarter' in df.columns:
            quarters = df['Quarter'].dropna().unique()
            stats['quarters_covered'] = quarters.tolist() if len(quarters) > 0 else []
        
        return stats
    
    def categorize_topic(self, text: str) -> str:
        """Simple topic categorization"""
        if pd.isna(text):
            return 'Other'
            
        text_lower = str(text).lower()
        
        # Simple keyword matching
        if any(word in text_lower for word in ['revenue', 'profit', 'earnings', 'income']):
            return 'Financial Performance'
        elif any(word in text_lower for word in ['risk', 'credit', 'loss', 'provision']):
            return 'Risk Management'
        elif any(word in text_lower for word in ['capital', 'tier', 'ratio', 'dividend']):
            return 'Capital'
        elif any(word in text_lower for word in ['strategy', 'plan', 'future', 'growth']):
            return 'Strategy'
        elif any(word in text_lower for word in ['market', 'economy', 'economic']):
            return 'Market Conditions'
        elif any(word in text_lower for word in ['cost', 'efficiency', 'operation']):
            return 'Operations'
        else:
            return 'Other'

# Simple utility functions
def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def extract_keywords(text: str) -> List[str]:
    """Extract basic keywords"""
    if pd.isna(text):
        return []
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', str(text).lower())
    
    # Remove common words
    stop_words = {'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'think', 'know', 'just', 'first', 'into', 'over', 'after', 'back', 'other', 'many', 'than', 'then', 'them', 'these', 'some', 'what', 'make', 'like', 'him', 'has', 'two'}
    
    keywords = [word for word in words if word not in stop_words]
    return list(set(keywords))[:10]  # Return top 10 unique keywords