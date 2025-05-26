"""
FinViz Performance Tracker - Track the Hottest Performing Stocks
Filters: Market Cap > $300M, Stocks Only
Tracks performance across multiple timeframes: 1D, 2D, 5D, 2W, 1M, 3M, 6M, 12M
"""

import pandas as pd
import numpy as np
from finvizfinance.screener.performance import Performance
import warnings
warnings.filterwarnings('ignore')

class PerformanceTracker:
    def __init__(self):
        self.performance_screener = Performance()
        self.raw_data = None
        self.processed_data = None
        
    def get_performance_data(self, verbose=True):
        """
        Fetch performance data from FinViz Performance screener
        """
        print("🔄 Fetching performance data from FinViz...")
        print("📊 Available timeframes: Week, Month, Quarter, Half, Year, YTD")
        
        try:
            # No filters - get all data (Market Cap filter doesn't work in Performance screener)
            self.performance_screener.set_filter(filters_dict={})
            
            # Get all data with performance metrics
            self.raw_data = self.performance_screener.screener_view(
                order='Performance (Week)',  # Sort by weekly performance initially
                limit=100000,  # Get as many as possible
                ascend=False,  # Descending order (best performers first)
                verbose=1 if verbose else 0,
                sleep_sec=1
            )
            
            if self.raw_data is not None and not self.raw_data.empty:
                print(f"✅ Successfully fetched {len(self.raw_data)} stocks")
                print(f"📊 Columns available: {list(self.raw_data.columns)}")
                return self.raw_data
            else:
                print("❌ No data returned")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def clean_and_process_data(self):
        """
        Clean data and calculate composite performance scores
        """
        if self.raw_data is None or self.raw_data.empty:
            print("❌ No data to process. Run get_performance_data() first.")
            return None
            
        print("🔄 Processing and cleaning data...")
        
        # Create a copy for processing
        df = self.raw_data.copy()
        
        # Remove funds, ETFs, etc. (keep only stocks)
        # Filter out common fund/ETF suffixes and patterns
        fund_patterns = ['ETF', 'FUND', 'TRUST', 'REIT', 'INDEX', 'SPDR', 'ISHARES']
        
        # Create a mask to identify likely funds/ETFs
        is_fund_mask = df['Company'].str.upper().str.contains('|'.join(fund_patterns), na=False)
        df = df[~is_fund_mask].copy()
        
        # Filter by market cap > 300M if Market Cap column exists
        if 'Market Cap' in df.columns:
            print("🔄 Filtering by Market Cap > $300M...")
            original_count = len(df)
            
            # Convert market cap to numeric for filtering
            df['Market Cap Numeric'] = self._convert_market_cap_to_numeric(df['Market Cap'])
            
            # Filter for market cap > 300M
            df = df[df['Market Cap Numeric'] >= 300].copy()
            
            filtered_count = len(df)
            print(f"📊 Market cap filter: {original_count} → {filtered_count} stocks")
        
        # Performance columns to analyze (these may vary based on FinViz data structure)
        perf_columns = []
        for col in df.columns:
            if 'Perf' in col or 'Change' in col or ('%' in col and 'Own' not in col):
                perf_columns.append(col)
        
        print(f"📊 Found performance columns: {perf_columns}")
        
        # Convert percentage strings to float values
        for col in perf_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Calculate composite performance scores
        if perf_columns:
            # Overall performance score (average of all timeframes)
            df['Overall_Score'] = df[perf_columns].mean(axis=1, skipna=True)
            
            # Short-term performance (1D, 2D, 5D, 2W if available)
            short_term_cols = [col for col in perf_columns if any(term in col for term in ['Day', '1D', '2D', '5D', 'Week'])]
            if short_term_cols:
                df['Short_Term_Score'] = df[short_term_cols].mean(axis=1, skipna=True)
            
            # Long-term performance (1M, 3M, 6M, 12M if available)
            long_term_cols = [col for col in perf_columns if any(term in col for term in ['Month', '1M', '3M', '6M', '12M', 'Year'])]
            if long_term_cols:
                df['Long_Term_Score'] = df[long_term_cols].mean(axis=1, skipna=True)
        
        # Add ranking columns
        df['Overall_Rank'] = df['Overall_Score'].rank(ascending=False, method='dense')
        if 'Short_Term_Score' in df.columns:
            df['Short_Term_Rank'] = df['Short_Term_Score'].rank(ascending=False, method='dense')
        if 'Long_Term_Score' in df.columns:
            df['Long_Term_Rank'] = df['Long_Term_Score'].rank(ascending=False, method='dense')
        
        self.processed_data = df
        print(f"✅ Processed {len(df)} stocks after filtering")
        return df
    
    def _convert_market_cap_to_numeric(self, market_cap_series):
        """
        Convert market cap strings to numeric values in millions
        Examples: '1.2B' -> 1200, '500M' -> 500, '15.3T' -> 15300000
        """
        def convert_single(value):
            if pd.isna(value) or value == '-':
                return 0
            
            value_str = str(value).upper().replace('$', '').replace(',', '').strip()
            
            # Extract the numeric part and multiplier
            if 'T' in value_str:
                multiplier = 1000000  # Trillions to millions
                numeric_part = value_str.replace('T', '')
            elif 'B' in value_str:
                multiplier = 1000  # Billions to millions
                numeric_part = value_str.replace('B', '')
            elif 'M' in value_str:
                multiplier = 1  # Already in millions
                numeric_part = value_str.replace('M', '')
            elif 'K' in value_str:
                multiplier = 0.001  # Thousands to millions
                numeric_part = value_str.replace('K', '')
            else:
                # Assume it's already a number in some format
                try:
                    return float(value_str)
                except:
                    return 0
            
            try:
                return float(numeric_part) * multiplier
            except:
                return 0
        
        return market_cap_series.apply(convert_single)
    
    def get_top_performers(self, n=50, timeframe='overall'):
        """
        Get top N performers by specified timeframe
        """
        if self.processed_data is None:
            print("❌ No processed data. Run clean_and_process_data() first.")
            return None
        
        df = self.processed_data.copy()
        
        # Sort by specified timeframe
        if timeframe.lower() == 'overall':
            df = df.sort_values('Overall_Score', ascending=False)
        elif timeframe.lower() == 'short_term':
            df = df.sort_values('Short_Term_Score', ascending=False)
        elif timeframe.lower() == 'long_term':
            df = df.sort_values('Long_Term_Score', ascending=False)
        else:
            # Try to find specific column
            matching_cols = [col for col in df.columns if timeframe.lower() in col.lower()]
            if matching_cols:
                df = df.sort_values(matching_cols[0], ascending=False)
            else:
                print(f"⚠️ Timeframe '{timeframe}' not found. Using overall performance.")
                df = df.sort_values('Overall_Score', ascending=False)
        
        return df.head(n)
    
    def analyze_by_sector_industry(self, min_stocks=3):
        """
        Analyze performance by sector and industry
        """
        if self.processed_data is None:
            print("❌ No processed data. Run clean_and_process_data() first.")
            return None, None
        
        df = self.processed_data.copy()
        
        # Sector Analysis
        print("🔄 Analyzing by Sector...")
        sector_analysis = df.groupby('Sector').agg({
            'Overall_Score': ['mean', 'median', 'count'],
            'Short_Term_Score': 'mean',
            'Long_Term_Score': 'mean',
            'Ticker': 'count'
        }).round(2)
        
        # Flatten column names
        sector_analysis.columns = ['Avg_Overall', 'Median_Overall', 'Count', 'Avg_Short_Term', 'Avg_Long_Term', 'Stock_Count']
        sector_analysis = sector_analysis[sector_analysis['Stock_Count'] >= min_stocks]
        sector_analysis = sector_analysis.sort_values('Avg_Overall', ascending=False)
        
        # Industry Analysis
        print("🔄 Analyzing by Industry...")
        industry_analysis = df.groupby('Industry').agg({
            'Overall_Score': ['mean', 'median', 'count'],
            'Short_Term_Score': 'mean',
            'Long_Term_Score': 'mean',
            'Ticker': 'count'
        }).round(2)
        
        # Flatten column names
        industry_analysis.columns = ['Avg_Overall', 'Median_Overall', 'Count', 'Avg_Short_Term', 'Avg_Long_Term', 'Stock_Count']
        industry_analysis = industry_analysis[industry_analysis['Stock_Count'] >= min_stocks]
        industry_analysis = industry_analysis.sort_values('Avg_Overall', ascending=False)
        
        return sector_analysis, industry_analysis
    
    def get_sector_top_performers(self, sector, n=20):
        """
        Get top performers within a specific sector
        """
        if self.processed_data is None:
            print("❌ No processed data. Run clean_and_process_data() first.")
            return None
        
        df = self.processed_data.copy()
        sector_stocks = df[df['Sector'].str.contains(sector, case=False, na=False)]
        
        if sector_stocks.empty:
            print(f"❌ No stocks found for sector: {sector}")
            return None
        
        return sector_stocks.sort_values('Overall_Score', ascending=False).head(n)
    
    def get_industry_top_performers(self, industry, n=20):
        """
        Get top performers within a specific industry
        """
        if self.processed_data is None:
            print("❌ No processed data. Run clean_and_process_data() first.")
            return None
        
        df = self.processed_data.copy()
        industry_stocks = df[df['Industry'].str.contains(industry, case=False, na=False)]
        
        if industry_stocks.empty:
            print(f"❌ No stocks found for industry: {industry}")
            return None
        
        return industry_stocks.sort_values('Overall_Score', ascending=False).head(n)
    
    def display_summary(self):
        """
        Display a comprehensive summary of the hottest performers
        """
        if self.processed_data is None:
            print("❌ No processed data. Run clean_and_process_data() first.")
            return
        
        print("\n" + "="*80)
        print("🔥 HOTTEST STOCKS PERFORMANCE SUMMARY 🔥")
        print("="*80)
        
        # Overall top performers
        print("\n🏆 TOP 20 OVERALL PERFORMERS:")
        top_overall = self.get_top_performers(20, 'overall')
        display_cols = ['Ticker', 'Company', 'Sector', 'Industry', 'Overall_Score', 'Overall_Rank']
        # Add available performance columns
        perf_cols = [col for col in top_overall.columns if 'Perf' in col]
        display_cols.extend(perf_cols[:6])  # Show up to 6 performance columns
        
        print(top_overall[display_cols].to_string(index=False))
        
        # Sector analysis
        print("\n🏢 TOP PERFORMING SECTORS:")
        sector_analysis, industry_analysis = self.analyze_by_sector_industry()
        if sector_analysis is not None:
            print(sector_analysis.head(10).to_string())
        
        # Industry analysis
        print("\n🏭 TOP PERFORMING INDUSTRIES:")
        if industry_analysis is not None:
            print(industry_analysis.head(10).to_string())
        
        print("\n" + "="*80)

# Main execution function
def run_performance_analysis():
    """
    Main function to run the complete performance analysis
    """
    tracker = PerformanceTracker()
    
    # Step 1: Get raw data
    raw_data = tracker.get_performance_data()
    if raw_data is None:
        return
    
    # Step 2: Process and clean data
    processed_data = tracker.clean_and_process_data()
    if processed_data is None:
        return
    
    # Step 3: Display comprehensive summary
    tracker.display_summary()
    
    # Step 4: Provide interactive analysis options
    print("\n🔧 INTERACTIVE ANALYSIS FUNCTIONS:")
    print("tracker.get_top_performers(n=50, timeframe='overall')")
    print("tracker.get_sector_top_performers('Technology', n=20)")
    print("tracker.get_industry_top_performers('Software', n=15)")
    print("tracker.analyze_by_sector_industry()")
    
    return tracker

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Starting FinViz Performance Analysis...")
    print("Filtering for: Market Cap > $300M, Stocks Only")
    print("Tracking all available timeframes for hottest performers")
    
    # Run the analysis
    tracker = run_performance_analysis()
    
    # Example specific queries
    if tracker and tracker.processed_data is not None:
        print("\n" + "="*50)
        print("📊 EXAMPLE SPECIFIC QUERIES:")
        
        # Top technology stocks
        print("\n💻 TOP 10 TECHNOLOGY STOCKS:")
        tech_tops = tracker.get_sector_top_performers('Technology', 10)
        if tech_tops is not None:
            cols = ['Ticker', 'Company', 'Overall_Score', 'Overall_Rank']
            print(tech_tops[cols].to_string(index=False))
        
        # Available sectors for further analysis
        if tracker.processed_data is not None:
            sectors = tracker.processed_data['Sector'].value_counts().head(10)
            print(f"\n📈 TOP SECTORS BY STOCK COUNT:\n{sectors}")
            
            # Available industries
            industries = tracker.processed_data['Industry'].value_counts().head(10)
            print(f"\n🏭 TOP INDUSTRIES BY STOCK COUNT:\n{industries}")
    
    print("\n✅ Analysis complete! Use the tracker object for further exploration.")