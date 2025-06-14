import pandas as pd
import re
from typing import List, Dict, Tuple
import openpyxl

def parse_earnings_transcript(transcript_text: str) -> Tuple[str, pd.DataFrame]:
    """
    Parse earnings call transcript to separate presentation from Q&A
    and structure Q&A into a dataframe.
    """
    
    # Split into presentation and Q&A sections
    qa_start = transcript_text.find("QUESTION AND ANSWER SECTION")
    
    if qa_start == -1:
        raise ValueError("Could not find Q&A section")
    
    presentation = transcript_text[:qa_start].strip()
    qa_section = transcript_text[qa_start:].strip()
    
    # Remove operator comments and disclaimer
    presentation = remove_operator_comments(presentation)
    qa_section = remove_operator_comments(qa_section)
    
    # Remove disclaimer section from Q&A
    disclaimer_start = qa_section.find("Disclaimer")
    if disclaimer_start != -1:
        qa_section = qa_section[:disclaimer_start].strip()
    
    # Parse Q&A into dataframe
    qa_df = parse_qa_section_new(qa_section)
    
    return presentation, qa_df

def remove_operator_comments(text: str) -> str:
    """Remove operator comments from text"""
    # Pattern to match operator comments (lines starting with "Operator:")
    operator_pattern = r'Operator:.*?(?=\n[A-Z]|$)'
    text = re.sub(operator_pattern, '', text, flags=re.DOTALL)
    
    # Remove excessive line breaks and dots
    text = re.sub(r'\.{10,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def parse_qa_section_new(qa_text: str) -> pd.DataFrame:
    """
    Simple and direct parsing approach for Q&A section
    """
    
    qa_data = []
    current_question_number = 0
    
    # Remove the Q&A section header
    qa_text = re.sub(r'QUESTION AND ANSWER SECTION.*?\n', '', qa_text, flags=re.DOTALL)
    
    # Split the text into lines
    lines = qa_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and operator comments
        if not line or line.startswith('Operator:'):
            i += 1
            continue
        
        # Look for speaker pattern: Name on one line, Title ending with Q or A on next line
        if i + 1 < len(lines):
            speaker_line = lines[i].strip()
            title_line = lines[i + 1].strip()
            
            # Check if title line ends with Q or A
            if title_line.endswith(' Q') or title_line.endswith(' A'):
                qa_marker = title_line[-1]
                company_title = title_line[:-2].strip()
                
                # Extract company
                company = extract_company_name(company_title)
                
                # Determine question number and type
                if qa_marker == 'Q':
                    current_question_number += 1
                    entry_type = 'Question'
                else:
                    entry_type = 'Answer'
                
                # Collect content until next speaker
                i += 2  # Skip speaker and title lines
                content_lines = []
                
                while i < len(lines):
                    current_line = lines[i].strip()
                    
                    # Stop if we hit another speaker (non-empty line followed by title with Q/A)
                    if (current_line and 
                        i + 1 < len(lines) and 
                        lines[i + 1].strip() and
                        (lines[i + 1].strip().endswith(' Q') or lines[i + 1].strip().endswith(' A'))):
                        break
                    
                    # Stop if we hit an operator comment
                    if current_line.startswith('Operator:'):
                        break
                    
                    if current_line:  # Only add non-empty lines
                        content_lines.append(current_line)
                    
                    i += 1
                
                # Clean and join content
                text_content = ' '.join(content_lines).strip()
                
                if text_content:
                    qa_data.append({
                        'question_number': current_question_number,
                        'type': entry_type,
                        'speaker': speaker_line,
                        'company': company,
                        'text': text_content
                    })
                
                continue
        
        i += 1
    
    return pd.DataFrame(qa_data)

def is_speaker_name(text: str) -> bool:
    """Check if text looks like a speaker name"""
    if not text or len(text) < 3:
        return False
    
    # Should start with capital letter and contain mostly letters/spaces
    if not text[0].isupper():
        return False
    
    # Should not contain common title words at the start
    title_words = ['analyst', 'chief', 'chairman', 'ceo', 'cfo', 'director', 'vice president']
    if any(text.lower().startswith(word) for word in title_words):
        return False
    
    # Should be mostly letters, spaces, and periods
    if re.match(r'^[A-Z][a-zA-Z\s\.\-\']+$', text):
        return True
    
    return False

def is_title_line(text: str) -> bool:
    """Check if text looks like a title/company line ending with Q or A"""
    if not text:
        return False
    
    # Should end with Q or A (possibly with space)
    if text.endswith(' Q') or text.endswith(' A') or text.endswith('Q') or text.endswith('A'):
        return True
    
    # Or contain typical title/company words
    title_indicators = ['analyst', 'chief', 'chairman', 'ceo', 'cfo', 'llc', 'inc', 'securities', 'bank', 'research']
    return any(indicator.lower() in text.lower() for indicator in title_indicators)

def extract_company_name(company_title: str) -> str:
    """Extract company name from title string"""
    
    # Remove Q/A marker first
    company_title = re.sub(r'\s*[QA]$', '', company_title).strip()
    
    # Common company mappings
    company_mappings = {
        "JPMorganChase": "JPMorgan Chase",
        "JPMorgan Chase": "JPMorgan Chase",
        "Autonomous": "Autonomous Research",
        'Autonomous Research': 'Autonomous Research',         
        "UBS": "UBS",
        'UBS Securities LLC': 'UBS',        
		"Deutsche Bank": "Deutsche Bank",
        "Wells Fargo" : "Wells Fargo Securities",
        'Wells Fargo Securities LLC': 'Wells Fargo Securities',        
		"Well Fargo": "Wells Fargo Securities",
        "Morgan Stanley": "Morgan Stanley",
        'Morgan Stanley & Co. LLC': 'Morgan Stanley',        
		"RBC": "RBC Capital Markets",
        'RBC Capital Markets LLC': 'RBC Capital Markets',        
		"Bank of America": "Bank of America",
        'Bank of America Merrill Lynch': 'Bank of America',        
		"Evercore":"Evercore",
        'Evercore Group LLC': 'Evercore',        
		"HSBC":"HSBC",
        'HSBC Securities (USA), Inc.': 'HSBC',        
		"Truist":"Truist Securities",
        "Wolfe":"Wolfe Research",
        "Seaport":"Seaport Global Securities",
        'Seaport Global Securities LLC': 'Seaport Global Securities',
        'Truist Securities Inc.': 'Truist Securities',
        'Deutsche Bank Securities, Inc.': 'Deutsche Bank',
        'Wolfe Research LLC': 'Wolfe Research'
    }
    
    # Check for exact matches first
    for key, value in company_mappings.items():
        if key in company_title:
            return value
        
    return company_title.strip()

def clean_text_content(text: str) -> str:
    """Clean up text content by removing excessive whitespace and dots"""
    # Remove lines with only dots
    text = re.sub(r'\n\.+\n', '\n', text)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def save_results(presentation: str, qa_df: pd.DataFrame, 
                presentation_file: str = 'presentation.txt',
                qa_file: str = 'q_and_a.xlsx'):
    """Save presentation and Q&A results to files"""
    
    # Save presentation to text file
    with open(presentation_file, 'w', encoding='utf-8') as f:
        f.write(presentation)
    
    # Save Q&A dataframe to Excel file
    with pd.ExcelWriter(qa_file, engine='openpyxl') as writer:
        qa_df.to_excel(writer, sheet_name='Q&A', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Q&A']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            # Set a reasonable max width
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Presentation saved to: {presentation_file}")
    print(f"Q&A dataframe saved to: {qa_file}")
    print(f"Q&A dataframe shape: {qa_df.shape}")
    
    # Display summary of Q&A data
    if not qa_df.empty:
        print(f"Number of questions: {qa_df[qa_df['type'] == 'Question'].shape[0]}")
        print(f"Number of answers: {qa_df[qa_df['type'] == 'Answer'].shape[0]}")
        print("\nQ&A Summary by Question:")
        summary = qa_df.groupby(['question_number', 'type']).size().unstack(fill_value=0)
        print(summary)
        
        print("\nSpeakers by Question:")
        for q_num in sorted(qa_df['question_number'].unique()):
            q_data = qa_df[qa_df['question_number'] == q_num]
            print(f"Question {q_num}: {list(q_data['speaker'].values)}")

def process_jpmorgan_transcript(full_transcript_text: str):
    """Process the actual JPMorgan transcript from the provided document"""
    
    try:
        print("Starting transcript processing...")
        print(f"Transcript length: {len(full_transcript_text)} characters")
        
        # Check if Q&A section exists
        qa_start = full_transcript_text.find("QUESTION AND ANSWER SECTION")
        print(f"Q&A section found at position: {qa_start}")
        
        if qa_start == -1:
            print("ERROR: Could not find Q&A section!")
            return None, None
        
        presentation, qa_df = parse_earnings_transcript(full_transcript_text)
        
        print(f"Parsing completed. DataFrame shape: {qa_df.shape}")
        if qa_df.empty:
            print("WARNING: DataFrame is empty!")
            
            # Debug the Q&A section
            qa_section = full_transcript_text[qa_start:qa_start+1000]
            print("First 1000 chars of Q&A section:")
            print(repr(qa_section))
            return None, None
        
        # Save results to Excel
        save_results(presentation, qa_df, 
                    'jpmorgan_presentation.txt', 
                    'q_and_a.xlsx')
        
        # Display first few rows for verification
        print("\n=== Q&A DataFrame ===")
        print(qa_df.to_string())
        
        return presentation, qa_df
        
    except Exception as e:
        print(f"Error processing transcript: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Extract the full transcript with the problematic Jamie Dimon sections
def extract_full_transcript():
    """Extract the complete transcript including all the Jamie Dimon interjections"""
    
    full_transcript = """1Q25 FINANCIAL RESULTS
EARNINGS CALL TRANSCRIPT
April 11, 2025

MANAGEMENT DISCUSSION SECTION

Operator: Good morning, ladies and gentlemen. Welcome to JPMorganChase's First-Quarter 2025 Earnings Call. This call is being recorded.

Jeremy Barnum
Chief Financial Officer, JPMorganChase

Thank you and good morning, everyone. Starting on page 1, the Firm reported net income of $14.6 billion, EPS of $5.07 on revenue of $46 billion, with an ROTCE of 21%.

QUESTION AND ANSWER SECTION

Operator: Thank you. Please stand by. Our first question comes from Ken Usdin with Autonomous. You may proceed.

Ken Usdin
Analyst, Autonomous Research Q

Good morning, Jeremy. Wondering if you could start by just kind of amplifying just the macro commentary that you started off on. And given the uncertainty in the world that you referenced, just how are you seeing the activity change across the customer base from consumers to wholesale? And can you just talk through how that's also just informing any changes in your – some of your growth and reserving expectations? Thanks.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

Sure, Ken. So I mean, at a high level, I would say that obviously, some of the salient news flow is quite recent. So, we've done some soundings and some checking both on the consumer side and on the wholesale side. I think on the consumer side, the thing to check is the spending data. And to be honest, the main thing that we see there, what would appear to be a certain amount of frontloading of spending ahead of people expecting price increases from tariffs. So ironically, that's actually somewhat supportive, all else equal. But I think what it sort of highlights is that during this transitional period and this elevated uncertainty, you might see some distortions in the data that make it hard to draw larger conclusions.

Ken Usdin
Analyst, Autonomous Research Q

Yeah. And just one question on the NII ex. Markets holding at $90 billion. Can you just walk us through the puts and takes of just what's the new curve you're using, which also is subject to change every day and what might have been some of the positive offsets to if you put in more expected cuts than you had before? Thanks.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

Yeah, that's a good question, Ken. You're right. So if you remember, last quarter we said that we had one cut in the curve. I think latest curve has something like three cuts. And so, we've talked a lot, obviously, about how we are asset sensitive. You now see our EaR disclosed in the supplement, and probably our empirical EaR is a little bit higher than our modeled EaR as a result of the relatively low – lower than modeled rates paid in Consumer. So when you put that together, all else equal, the drop in the weighted-average IORB, which is about 22 basis points, should produce a notable headwind in our NII ex. Markets.

Jamie Dimon
Chairman & Chief Executive Officer, JPMorganChase A

In the curve basically.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

Yeah. This is basically – that's just mechanically...

Jamie Dimon
Chairman & Chief Executive Officer, JPMorganChase A

This is guaranteed not to happen.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

So that's mechanically just flowing through the curve. So yeah, your question is that given that, why are you not revising down? And the answer to that is that across all the puts and takes actually, our number is a tiny bit lower. It's just not enough to warrant a change in the outlook. But we do have some offsets. So we have some balance effects that are favorable. You will have noted that I talked about higher wholesale deposit balances, for example. We see beta outperforming in a couple of different places in CDs and in wholesale.

Operator: Thank you. Our next question comes from Erika Najarian with UBS. You may proceed.

Erika Najarian
Analyst, UBS Securities LLC Q

Yes. Good morning. This question is for Jamie. Jamie, you were on the media today talking about potential economic turbulence. But Jeremy also mentioned that banking should be – the banking system should be a source of strength in this turbulence. The equity market always seems to think about the banks as weaker players given how they trade the stocks more on sentiment and fear rather than the math of – the ability of banks to absorb provisions going forward, if we do fall into a slower economic downturn.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

I just – before Jamie answers that, Erika, I just want to make one brief comment, which is the banking system being a source of strength means what it says. In other words, banks doing their job to support the economy. That's not a statement about bank equity performance and the extent to which banks are cyclical or not.

Jamie Dimon
Chairman & Chief Executive Officer, JPMorganChase A

Everyone trades stocks in a different way, so sentiment – but banks are a cork in the ocean when it comes to the economy. If the economy gets worse, credit losses will go up, volumes can change, deal curves can change. And we're not predicting all of that. What I would say is our excellent economist, Michael Feroli, I called him this morning, specifically to ask him, how they're looking at their forecast today, they think it's about 50-50 for a recession."""
    
    return full_transcript

# Test the parsing with a simple example
def test_parsing():
    """Test the parsing with a simple example"""
    
    test_transcript = """MANAGEMENT DISCUSSION SECTION

Jeremy Barnum
Chief Financial Officer, JPMorganChase

Thank you and good morning, everyone.

QUESTION AND ANSWER SECTION

Ken Usdin
Analyst, Autonomous Research Q

Good morning, Jeremy. This is my question.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

Sure, Ken. This is my answer.

Jamie Dimon
Chairman & Chief Executive Officer, JPMorganChase A

Short comment.

Jeremy Barnum
Chief Financial Officer, JPMorganChase A

More answer content here."""
    
    print("=== TESTING PARSING ===")
    presentation, qa_df = parse_earnings_transcript(test_transcript)
    
    print(f"Test result - DataFrame shape: {qa_df.shape}")
    if not qa_df.empty:
        print("Test DataFrame:")
        print(qa_df.to_string())
    else:
        print("Test failed - empty DataFrame")
    
    return qa_df

# Process the actual JPMorgan transcript
if __name__ == "__main__":
    print("Testing parsing logic first...")
    test_df = test_parsing()
    
    if test_df is not None and not test_df.empty:
        print("\n" + "="*50)
        print("Test passed! Now processing full transcript...")
        
        # Get the full transcript
        full_text = extract_full_transcript()
        
        # Process it
        presentation, qa_df = process_jpmorgan_transcript(full_text)
        
        if qa_df is not None and not qa_df.empty:
            print("\n=== SUCCESS: Transcript processed successfully! ===")
            print(f"Generated {qa_df.shape[0]} Q&A entries")
            print(f"Questions: {qa_df[qa_df['type'] == 'Question'].shape[0]}")
            print(f"Answers: {qa_df[qa_df['type'] == 'Answer'].shape[0]}")
        else:
            print("ERROR: Failed to process full transcript")
    else:
        print("ERROR: Basic test failed")