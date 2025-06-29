import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json, os, sys, time

class AlphaVantageEarnings:
    def __init__(self, api_key):
        """
        Initialize with your Alpha Vantage API key
        Get your free API key from: https://www.alphavantage.co/support/#api-key
        """
        #api_key = "MD95HJVCFVUWIPHG"
        api_key = "FU4V5XSIS7W5AUDE"
        if not api_key:
            raise Exception("API key is required")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_earnings_data(self, symbol="JPM"):
        """
        Fetch quarterly earnings data for JP Morgan (JPM)
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                raise Exception(f"API Limit: {data['Note']}")
                
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def process_quarterly_data(self, earnings_data):
        """
        Process and clean the quarterly earnings data
        """
        if 'quarterlyEarnings' not in earnings_data:
            raise Exception("No quarterly earnings data found")
        
        quarterly = earnings_data['quarterlyEarnings']
        
        # Convert to DataFrame
        df = pd.DataFrame(quarterly)
        
        # Convert string values to numeric where possible
        numeric_columns = ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date column
        if 'fiscalDateEnding' in df.columns:
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('fiscalDateEnding', ascending=False)
        
        return df
    
    def display_earnings_summary(self, df, num_quarters=8):
        """
        Display a formatted summary of recent quarterly earnings
        """
        print("=" * 80)
        print(f"JP MORGAN CHASE & CO. (JPM) - QUARTERLY EARNINGS SUMMARY")
        print("=" * 80)
        print(f"Data Source: Alpha Vantage API")
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # Display recent quarters
        recent_data = df.head(num_quarters)
        
        for _, row in recent_data.iterrows():
            print(f"\nQuarter Ending: {row['fiscalDateEnding'].strftime('%Y-%m-%d')}")
            print(f"Reported Date: {row.get('reportedDate', 'N/A')}")
            print(f"Reported EPS: ${row.get('reportedEPS', 'N/A'):.2f}" if pd.notna(row.get('reportedEPS')) else "Reported EPS: N/A")
            print(f"Estimated EPS: ${row.get('estimatedEPS', 'N/A'):.2f}" if pd.notna(row.get('estimatedEPS')) else "Estimated EPS: N/A")
            
            if pd.notna(row.get('surprise')):
                surprise_symbol = "📈" if row['surprise'] > 0 else "📉" if row['surprise'] < 0 else "➡️"
                print(f"Surprise: ${row['surprise']:.2f} {surprise_symbol}")
            
            if pd.notna(row.get('surprisePercentage')):
                print(f"Surprise %: {row['surprisePercentage']:.2f}%")
            
            print("-" * 40)
    
    def create_earnings_visualization(self, df):
        """
        Create visualizations of the earnings data
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('JP Morgan Chase & Co. (JPM) - Quarterly Earnings Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Prepare data for plotting (last 12 quarters)
        plot_data = df.head(12).copy()
        plot_data = plot_data.sort_values('fiscalDateEnding')
        
        # 1. EPS Trend Over Time
        axes[0, 0].plot(plot_data['fiscalDateEnding'], plot_data['reportedEPS'], 
                       marker='o', linewidth=2, markersize=6, color='#1f77b4', label='Reported EPS')
        axes[0, 0].plot(plot_data['fiscalDateEnding'], plot_data['estimatedEPS'], 
                       marker='s', linewidth=2, markersize=6, color='#ff7f0e', 
                       linestyle='--', label='Estimated EPS')
        axes[0, 0].set_title('Earnings Per Share Trend', fontweight='bold')
        axes[0, 0].set_xlabel('Quarter')
        axes[0, 0].set_ylabel('EPS ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Earnings Surprise
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                 for x in plot_data['surprise'].fillna(0)]
        bars = axes[0, 1].bar(range(len(plot_data)), plot_data['surprise'].fillna(0), 
                             color=colors, alpha=0.7)
        axes[0, 1].set_title('Earnings Surprise by Quarter', fontweight='bold')
        axes[0, 1].set_xlabel('Quarter (Most Recent to Oldest)')
        axes[0, 1].set_ylabel('Surprise ($)')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if abs(height) > 0.01:  # Only show labels for non-zero values
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:.2f}', ha='center', 
                               va='bottom' if height > 0 else 'top')
        
        # 3. Surprise Percentage
        colors_pct = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                     for x in plot_data['surprisePercentage'].fillna(0)]
        axes[1, 0].bar(range(len(plot_data)), plot_data['surprisePercentage'].fillna(0), 
                      color=colors_pct, alpha=0.7)
        axes[1, 0].set_title('Earnings Surprise Percentage', fontweight='bold')
        axes[1, 0].set_xlabel('Quarter (Most Recent to Oldest)')
        axes[1, 0].set_ylabel('Surprise (%)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        axes[1, 1].axis('off')
        
        # Calculate summary statistics
        avg_reported = plot_data['reportedEPS'].mean()
        avg_surprise = plot_data['surprise'].mean()
        avg_surprise_pct = plot_data['surprisePercentage'].mean()
        beat_rate = (plot_data['surprise'] > 0).sum() / len(plot_data) * 100
        
        summary_text = f"""
        EARNINGS SUMMARY (Last 12 Quarters)
        ────────────────────────────────────
        
        Average Reported EPS: ${avg_reported:.2f}
        
        Average Surprise: ${avg_surprise:.2f}
        
        Average Surprise %: {avg_surprise_pct:.1f}%
        
        Beat Rate: {beat_rate:.0f}%
        
        Latest Quarter EPS: ${plot_data.iloc[-1]['reportedEPS']:.2f}
        
        Data Source: Alpha Vantage API
        Generated: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def export_to_csv(self, df, filename="jpmorgan_quarterly_earnings.csv"):
        """
        Export the earnings data to CSV
        """
        df.to_csv(filename, index=False)
        print(f"\nData exported to: {filename}")

# Example usage and main execution
def main():
    # You need to get your API key from Alpha Vantage
    API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')  # Replace with your actual API key
    
    try:
        # Initialize the earnings analyzer
        earnings = AlphaVantageEarnings(API_KEY)
        
        print("Fetching JP Morgan quarterly earnings data...")
        
        # Get the earnings data
        raw_data = earnings.get_earnings_data("JPM")
        
        # Process the data
        df = earnings.process_quarterly_data(raw_data)
        
        # Display summary
        earnings.display_earnings_summary(df)
        
        # Create visualizations
        earnings.create_earnings_visualization(df)
        
        # Export to CSV
        earnings.export_to_csv(df)
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a valid Alpha Vantage API key")
        print("2. Check your internet connection")
        print("3. Verify you haven't exceeded API rate limits (5 calls/minute for free tier)")

if __name__ == "__main__":
    # Required packages
    print("Required packages: requests, pandas, matplotlib, seaborn")
    print("Install with: pip install requests pandas matplotlib seaborn")
    print("\n" + "="*50)
    
    main()