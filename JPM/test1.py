def demonstrate_operator_removal():
    """
    Demonstrate the removal of operator comments and dotted lines
    while preserving speaker details.
    """
    
    # Sample text showing the exact pattern from JPMorgan transcript
    sample_text = """1Q25 FINANCIAL RESULTS 
EARNINGS CALL TRANSCRIPT 
April 11, 2025 

MANAGEMENT DISCUSSION SECTION 

Operator: Good morning, ladies and gentlemen. Welcome to JPMorganChase's First-Quarter 2025 Earnings Call. This call is being recorded. 
Your line will be muted for the duration of the call. We will now go live to the presentation. The presentation is available on JPMorganChase's 
website. Please refer to the disclaimer in the back concerning forward-looking statements. Please stand by. 

At this time, I would like to turn the call over to JPMorganChase's Chairman and CEO, Jamie Dimon, and the Chief Financial Officer, Jeremy 
Barnum. Mr. Barnum, please go ahead. 
........................................................................................................................................................................................................................................................................................ 

Jeremy Barnum 
Chief Financial Officer, JPMorganChase 
Thank you and good morning, everyone. Starting on page 1, the Firm reported net income of $14.6 billion, EPS of $5.07 on revenue of $46 
billion, with an ROTCE of 21%. These results included a First Republic-related gain of $588 million, which was previously disclosed in the 10-K.

On page 2, we have more on our first-quarter results. The Firm reported revenue of $46 billion, up $3.5 billion, or 8% year-on-year.

QUESTION AND ANSWER SECTION 
 
Operator: Thank you. Please stand by. Our first question comes from Ken Usdin with Autonomous. You may proceed. 
........................................................................................................................................................................................................................................................................................ 

Ken Usdin 
Analyst, Autonomous Research Q
Good morning, Jeremy. Wondering if you could start by just kind of amplifying just the macro commentary that you started off on.

........................................................................................................................................................................................................................................................................................ 

Jeremy Barnum 
Chief Financial Officer, JPMorganChase A
Sure, Ken. So I mean, at a high level, I would say that obviously, some of the salient news flow is quite recent.

........................................................................................................................................................................................................................................................................................ 

Operator: Thank you. Our next question comes from Erika Najarian with UBS. You may proceed. 
........................................................................................................................................................................................................................................................................................ 

Erika Najarian 
Analyst, UBS Securities LLC Q
Yes. Good morning. This question is for Jamie. Jamie, you were on the media today talking about potential economic turbulence.

........................................................................................................................................................................................................................................................................................ 

Jeremy Barnum 
Chief Financial Officer, JPMorganChase A
I just – before Jamie answers that, Erika, I just want to make one brief comment.

........................................................................................................................................................................................................................................................................................ 

Jamie Dimon 
Chairman & Chief Executive Officer, JPMorganChase A
Everyone trades stocks in a different way, so sentiment – but banks are a cork in the ocean when it comes to the economy.

........................................................................................................................................................................................................................................................................................ 

Operator: Thank you. Our next question comes from Matt O'Connor with Deutsche Bank. Your line is open. 
........................................................................................................................................................................................................................................................................................ 

Matt O'Connor 
Analyst, Deutsche Bank Securities, Inc. Q
Good morning. Just wanted to drill down on the credit card spend. Any comments in terms of changing patterns on the consumer card spend?"""

    def clean_transcript_simple(text):
        """Simple cleaning that removes operator + dots but keeps speaker details"""
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Check if this is an operator line
            if line_stripped.startswith("Operator:"):
                # Skip the operator line and continue reading until we find dotted divider
                i += 1
                
                # Keep skipping lines until we find the dotted divider or next speaker
                while i < len(lines):
                    current_line = lines[i].strip()
                    
                    # If we find the dotted divider, skip it and break
                    if current_line.startswith(".") and len(current_line) > 10:
                        i += 1  # Skip the dotted line
                        break
                    # If we find a speaker name (next section), don't skip it
                    elif any(name in current_line for name in ["Jeremy Barnum", "Jamie Dimon"]) and any(title in current_line for title in ["Chief Financial Officer", "Chairman"]):
                        break
                    # If we find an analyst line, don't skip it  
                    elif current_line.endswith(" Q") and "Analyst," in current_line:
                        break
                    # If it's just a name line (like "Ken Usdin"), don't skip it
                    elif current_line and not current_line.startswith(".") and len(current_line.split()) <= 3 and current_line[0].isupper():
                        break
                    # Otherwise, this is part of the operator comment, so skip it
                    else:
                        i += 1
                
                continue
            
            # Check if this is a standalone divider line (dots only)
            elif line_stripped.startswith(".") and len(line_stripped) > 20:
                # Skip standalone divider lines
                i += 1
                continue
            
            else:
                # Keep all other lines (including speaker details, content, etc.)
                cleaned_lines.append(line)
                i += 1
        
        return '\n'.join(cleaned_lines)
    
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(sample_text[:1000] + "..." if len(sample_text) > 1000 else sample_text)
    print("\n" + "=" * 80)
    
    # Clean the text
    cleaned = clean_transcript_simple(sample_text)
    
    print("\nCLEANED TEXT:")
    print("=" * 80)
    print(cleaned)
    print("\n" + "=" * 80)
    
    # Show what was removed
    original_lines = sample_text.split('\n')
    cleaned_lines = cleaned.split('\n')
    
    print(f"\nSTATISTICS:")
    print(f"Original lines: {len(original_lines)}")
    print(f"Cleaned lines: {len(cleaned_lines)}")
    print(f"Lines removed: {len(original_lines) - len(cleaned_lines)}")
    
    # Show removed content
    print(f"\nREMOVED CONTENT:")
    print("-" * 40)
    
    # Find operator lines in original
    for line in original_lines:
        if line.strip().startswith("Operator:"):
            print(f"REMOVED: {line.strip()}")
        elif line.strip().startswith(".") and len(line.strip()) > 20:
            print(f"REMOVED: [Dotted divider line]")
    
    # Show preserved speaker details
    print(f"\nPRESERVED SPEAKER DETAILS:")
    print("-" * 40)
    
    in_speaker_block = False
    for line in cleaned_lines:
        stripped = line.strip()
        if any(title in stripped for title in ["Chief Financial Officer", "Chairman & Chief Executive Officer", "Analyst,"]):
            print(f"KEPT: {stripped}")
        elif stripped.endswith(" Q") or stripped.endswith(" A"):
            print(f"KEPT: {stripped}")
    
    return cleaned

if __name__ == "__main__":
    demonstrate_operator_removal()