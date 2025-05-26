"""
Industry Stock Screener - Find Strongest Names in Each Industry
Full-scale version with proper filtering and rate limiting
Filters: Market Cap (Small+), Price above SMA50, Price above SMA200
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
import random
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# Import finvizfinance screener modules
try:
    from finvizfinance.screener.overview import Overview
    from finvizfinance.screener.performance import Performance
    from finvizfinance.screener.technical import Technical
    SCREENER_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Screener modules not available: {e}")
    SCREENER_MODULES_AVAILABLE = False

class IndustryStockScreener:
    def __init__(self):
        """Initialize the Industry Stock Screener"""
        self.overview_screener = Overview()
        self.performance_screener = Performance()
        self.technical_screener = Technical()
        self.industry_results = {}
        
        # Rate limiting settings for API protection
        self.base_delay = 2.0  # Base delay between requests
        self.max_delay = 10.0  # Maximum delay
        self.retry_delay = 5.0  # Delay on retry
        self.max_retries = 5   # Maximum retries per request
        
        # Columns to exclude (as specified by user)
        self.excluded_columns = {
            'Beta', 'SMA20', 'SMA50', 'SMA200', '52W High', '52W Low', 'RSI', 
            'Change', 'Gap', 'Volume', 'Change from Open', 'Recom', 'Avg Volume', 
            'Rel Volume', 'Market Cap', 'P/E', 'Overall_Strength', 'Short_Term_Strength', 
            'Long_Term_Strength', 'SMA_Strength', 'Rank', 'Change From Open',
            'Average Volume (3 Month)', 'Relative Volume', 'Market Cap.',
            'Price/Earnings', 'Analyst Recommendation', 'Average True Range',
            'Relative Strength Index (14)', '20-Day SMA (Relative)', 
            '50-Day SMA (Relative)', '200-Day SMA (Relative)', '52-Week High (Relative)',
            '52-Week Low (Relative)', 'Volatility (Week)', 'Volatility (Month)'
        }
        
        print("🔍 INDUSTRY STOCK SCREENER - FULL SCALE")
        print("="*60)
        print("📊 Filters: Market Cap (Small+), Price > SMA50, Price > SMA200")
        print("📊 Processing ALL industries with rate limiting")
        print("="*60)
        
        if SCREENER_MODULES_AVAILABLE:
            self.setup_filters()
        else:
            print("❌ Screener modules not available")

    def setup_filters(self):
        """Setup the correct filter combinations"""
        print("\n🔧 Setting up filters...")
        
        # Based on the documentation and your requirements
        self.core_filters = {
            'Market Cap.': 'Small+ (over $300M)',  # Try this format first
            '200-Day Simple Moving Average': 'Price above SMA200',
            '50-Day Simple Moving Average': 'Price above SMA50'
        }
        
        # Alternative market cap formats to try
        self.market_cap_alternatives = [
            'Small+ (over $300M)',
            '+Small (over $300M)', 
            'Small+',
            '+Small',
            'Small ($300M to $2B)',
            'Over $300M'
        ]
        
        # Test the filters
        self.working_filters = self.test_filter_combinations()
        
    def test_filter_combinations(self):
        """Test different filter combinations to find what works"""
        print("🧪 Testing filter combinations...")
        
        working_filters = {}
        
        # Test SMA filters first (we know these work)
        sma_filters = {
            '200-Day Simple Moving Average': 'Price above SMA200',
            '50-Day Simple Moving Average': 'Price above SMA50'
        }
        
        try:
            self.technical_screener.set_filter(filters_dict=sma_filters)
            result = self.technical_screener.screener_view(limit=1, verbose=0)
            if result is not None and not result.empty:
                working_filters.update(sma_filters)
                print("✅ SMA filters confirmed working")
        except Exception as e:
            print(f"❌ SMA filters failed: {e}")
        
        # Test market cap filters
        for market_cap_option in self.market_cap_alternatives:
            try:
                test_filters = {'Market Cap.': market_cap_option}
                test_filters.update(sma_filters)  # Include SMA filters
                
                self.technical_screener.set_filter(filters_dict=test_filters)
                result = self.technical_screener.screener_view(limit=1, verbose=0)
                
                if result is not None and not result.empty:
                    working_filters['Market Cap.'] = market_cap_option
                    print(f"✅ Market Cap filter working: {market_cap_option}")
                    break
                    
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Market Cap '{market_cap_option}' failed: {str(e)[:50]}")
                continue
        
        if 'Market Cap.' not in working_filters:
            print("⚠️ No market cap filter found - proceeding with SMA filters only")
        
        print(f"🔧 Working filters: {working_filters}")
        return working_filters

    def smart_delay(self, base_delay: float = None):
        """Implement smart delay with randomization to avoid API limits"""
        if base_delay is None:
            base_delay = self.base_delay
        
        # Add randomization to avoid synchronized requests
        delay = base_delay + random.uniform(0.5, 2.0)
        time.sleep(delay)

    def exponential_backoff_delay(self, attempt: int):
        """Exponential backoff for retries"""
        delay = min(self.retry_delay * (2 ** attempt), self.max_delay)
        delay += random.uniform(0.5, 1.5)  # Add jitter
        time.sleep(delay)

    def clean_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove excluded columns from dataframe"""
        if df is None or df.empty:
            return df
        
        # Remove excluded columns
        columns_to_drop = [col for col in df.columns if col in self.excluded_columns]
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"      🧹 Removed {len(columns_to_drop)} excluded columns")
        
        return df

    def screen_industry_with_retries(self, industry_name: str) -> Optional[pd.DataFrame]:
        """Screen a single industry with comprehensive retry logic"""
        print(f"\n🔍 Screening: {industry_name}")
        
        # Build filters for this industry
        industry_filters = self.working_filters.copy()
        industry_filters['Industry'] = industry_name
        
        print(f"📊 Filters: {industry_filters}")
        
        # Try multiple screeners to get comprehensive data
        all_results = []
        
        # 1. Technical Screener (has SMA filters)
        tech_data = self.fetch_with_retries(
            self.technical_screener, 
            industry_filters, 
            'Performance (Week)',
            f"{industry_name} - Technical"
        )
        if tech_data is not None:
            tech_data = self.clean_dataframe_columns(tech_data)
            all_results.append(('Technical', tech_data))
        
        # 2. Performance Screener (has performance data)
        perf_filters = {k: v for k, v in industry_filters.items() 
                       if 'SMA' not in k and 'Moving Average' not in k}
        
        if perf_filters:
            perf_data = self.fetch_with_retries(
                self.performance_screener,
                perf_filters,
                'Performance (Month)',
                f"{industry_name} - Performance"
            )
            if perf_data is not None:
                perf_data = self.clean_dataframe_columns(perf_data)
                all_results.append(('Performance', perf_data))
        
        # 3. Overview Screener (has fundamental data)
        overview_filters = {k: v for k, v in industry_filters.items() 
                           if 'SMA' not in k and 'Moving Average' not in k}
        
        if overview_filters:
            overview_data = self.fetch_with_retries(
                self.overview_screener,
                overview_filters,
                'Market Cap.',
                f"{industry_name} - Overview"
            )
            if overview_data is not None:
                overview_data = self.clean_dataframe_columns(overview_data)
                all_results.append(('Overview', overview_data))
        
        # Merge all results
        if all_results:
            merged_data = self.merge_screener_results(all_results)
            if merged_data is not None and not merged_data.empty:
                print(f"✅ {industry_name}: {len(merged_data)} stocks")
                return merged_data
        
        print(f"❌ {industry_name}: No data found")
        return None

    def fetch_with_retries(self, screener, filters_dict: Dict, order_by: str, context: str) -> Optional[pd.DataFrame]:
        """Fetch data with comprehensive retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                # Set filters
                screener.set_filter(filters_dict=filters_dict)
                
                # Add delay before request
                if attempt > 0:
                    self.exponential_backoff_delay(attempt)
                else:
                    self.smart_delay()
                
                # Make request
                result = screener.screener_view(
                    order=order_by,
                    limit=200,  # Get more stocks per industry
                    ascend=False,
                    verbose=0,
                    sleep_sec=1
                )
                
                if result is not None and not result.empty:
                    print(f"      ✅ {context}: {len(result)} stocks")
                    return result
                else:
                    print(f"      ⚠️ {context}: No data (attempt {attempt + 1})")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"      ❌ {context} attempt {attempt + 1}: {error_msg[:100]}")
                
                # Check for rate limiting indicators
                if any(indicator in error_msg.lower() for indicator in ['rate', 'limit', 'quota', 'throttle']):
                    print(f"      🐌 Rate limit detected, increasing delay...")
                    self.exponential_backoff_delay(attempt + 2)  # Extra delay for rate limits
                elif attempt < self.max_retries - 1:
                    self.exponential_backoff_delay(attempt)
        
        print(f"      ❌ {context}: Failed after {self.max_retries} attempts")
        return None

    def merge_screener_results(self, screener_results: List) -> Optional[pd.DataFrame]:
        """Merge results from multiple screeners"""
        if not screener_results:
            return None
        
        # Start with the largest dataset
        screener_results.sort(key=lambda x: len(x[1]), reverse=True)
        
        merged_data = screener_results[0][1].copy()
        source_name = screener_results[0][0]
        
        print(f"      📊 Base: {source_name} ({len(merged_data)} stocks)")
        
        # Merge additional datasets
        for source_name, data in screener_results[1:]:
            if data is not None and not data.empty:
                print(f"      📊 Merging: {source_name} ({len(data)} stocks)")
                
                # Merge on Ticker with outer join to get all stocks
                merged_data = pd.merge(
                    merged_data, 
                    data, 
                    on='Ticker', 
                    how='outer',
                    suffixes=('', f'_{source_name}')
                )
        
        # Clean up duplicate columns (keep non-suffixed versions)
        columns_to_drop = []
        for col in merged_data.columns:
            if '_' in col and any(col.endswith(f'_{suffix}') for suffix in ['Technical', 'Performance', 'Overview']):
                base_col = col.split('_')[0]
                if base_col in merged_data.columns:
                    columns_to_drop.append(col)
        
        if columns_to_drop:
            merged_data = merged_data.drop(columns=columns_to_drop)
            print(f"      🧹 Removed {len(columns_to_drop)} duplicate columns")
        
        print(f"      ✅ Final merged: {len(merged_data)} stocks, {len(merged_data.columns)} columns")
        return merged_data

    def process_all_industries(self, industries_list: List[str], batch_size: int = 5):
        """Process all industries in batches with comprehensive rate limiting"""
        print(f"\n🚀 Processing {len(industries_list)} industries in batches of {batch_size}")
        
        self.industry_results = {}
        total_stocks = 0
        successful_industries = 0
        
        # Process in batches
        for i in range(0, len(industries_list), batch_size):
            batch = industries_list[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(industries_list) + batch_size - 1) // batch_size
            
            print(f"\n📦 BATCH {batch_num}/{total_batches}: {len(batch)} industries")
            print(f"📊 Progress: {i}/{len(industries_list)} industries completed")
            
            # Process batch
            for j, industry in enumerate(batch):
                try:
                    print(f"\n🏢 Industry {i+j+1}/{len(industries_list)}: {industry}")
                    
                    # Screen the industry
                    industry_data = self.screen_industry_with_retries(industry)
                    
                    if industry_data is not None and not industry_data.empty:
                        self.industry_results[industry] = industry_data
                        successful_industries += 1
                        total_stocks += len(industry_data)
                        
                        print(f"✅ Stored: {industry} ({len(industry_data)} stocks)")
                    else:
                        print(f"❌ No data: {industry}")
                    
                    # Rate limiting between industries
                    self.smart_delay()
                    
                except Exception as e:
                    print(f"❌ Error processing {industry}: {e}")
                    continue
            
            # Longer delay between batches
            if i + batch_size < len(industries_list):
                print(f"\n⏸️ Batch complete. Waiting before next batch...")
                time.sleep(random.uniform(5.0, 10.0))
        
        print(f"\n✅ ALL INDUSTRIES PROCESSED!")
        print(f"📊 Success rate: {successful_industries}/{len(industries_list)} industries")
        print(f"📊 Total stocks found: {total_stocks}")
        
        return self.industry_results

    def display_summary_results(self, top_per_industry: int = 5):
        """Display comprehensive summary of all results"""
        if not self.industry_results:
            print("❌ No results to display")
            return
        
        print(f"\n🌟 INDUSTRY SCREENING RESULTS SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_stocks = sum(len(data) for data in self.industry_results.values() if data is not None)
        total_industries = len(self.industry_results)
        
        print(f"📊 Industries processed: {total_industries}")
        print(f"📊 Total stocks found: {total_stocks}")
        print(f"📊 Average stocks per industry: {total_stocks/total_industries:.1f}")
        
        # Top industries by stock count
        industry_counts = [(industry, len(data)) for industry, data in self.industry_results.items() 
                          if data is not None and not data.empty]
        industry_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 10 INDUSTRIES BY STOCK COUNT:")
        for industry, count in industry_counts[:10]:
            print(f"  {industry}: {count} stocks")
        
        # Display top stocks per industry
        print(f"\n🌟 TOP {top_per_industry} STOCKS FROM EACH INDUSTRY")
        print("="*80)
        
        for industry, data in self.industry_results.items():
            if data is not None and not data.empty:
                print(f"\n🏢 {industry.upper()} (Top {min(top_per_industry, len(data))})")
                print("-" * 60)
                
                # Sort by performance if available, otherwise by price
                if 'Performance (Month)' in data.columns:
                    top_stocks = data.nlargest(top_per_industry, 'Performance (Month)')
                elif 'Performance (Week)' in data.columns:
                    top_stocks = data.nlargest(top_per_industry, 'Performance (Week)')
                else:
                    top_stocks = data.head(top_per_industry)
                
                for idx, (_, stock) in enumerate(top_stocks.iterrows(), 1):
                    ticker = stock['Ticker']
                    company = stock.get('Company', 'N/A')
                    if len(str(company)) > 30:
                        company = str(company)[:30]
                    
                    price = stock.get('Price', 0)
                    
                    # Get best available performance metric
                    perf_value = 0
                    perf_label = "N/A"
                    
                    for perf_col in ['Performance (Month)', 'Performance (Week)', 'Performance (Quarter)']:
                        if perf_col in stock and pd.notna(stock[perf_col]):
                            perf_value = stock[perf_col]
                            perf_label = perf_col.replace('Performance (', '').replace(')', '')
                            break
                    
                    print(f"  {idx}. {ticker:6} | {company:30} | ${price:6.2f} | {perf_label}: {perf_value:+6.2f}%")

    def export_comprehensive_results(self, filename: str = None):
        """Export all results to a comprehensive Excel file"""
        if not self.industry_results:
            print("❌ No results to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"industry_screener_full_results_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                all_stocks_data = []
                
                for industry, data in self.industry_results.items():
                    if data is not None and not data.empty:
                        # Add to summary
                        avg_performance = 0
                        if 'Performance (Month)' in data.columns:
                            avg_performance = data['Performance (Month)'].mean()
                        
                        summary_data.append({
                            'Industry': industry,
                            'Stock_Count': len(data),
                            'Top_Stock': data.iloc[0]['Ticker'] if len(data) > 0 else 'N/A',
                            'Avg_Monthly_Performance': avg_performance
                        })
                        
                        # Add industry column and append to all stocks
                        data_copy = data.copy()
                        data_copy['Industry_Name'] = industry
                        all_stocks_data.append(data_copy)
                
                # Summary sheet
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df.sort_values('Stock_Count', ascending=False)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # All stocks combined sheet
                if all_stocks_data:
                    all_stocks_df = pd.concat(all_stocks_data, ignore_index=True)
                    all_stocks_df.to_excel(writer, sheet_name='All_Stocks', index=False)
                
                # Individual industry sheets (top 20 industries by stock count)
                industry_counts = [(industry, len(data)) for industry, data in self.industry_results.items() 
                                 if data is not None and not data.empty]
                industry_counts.sort(key=lambda x: x[1], reverse=True)
                
                for industry, count in industry_counts[:20]:  # Top 20 industries
                    data = self.industry_results[industry]
                    if data is not None and not data.empty:
                        # Clean sheet name for Excel compatibility
                        sheet_name = industry.replace('/', '_').replace('\\', '_').replace('&', 'and')[:31]
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ Comprehensive results exported to: {filename}")
            
            # Print export summary
            total_stocks = sum(len(data) for data in self.industry_results.values() if data is not None)
            print(f"📊 Export summary:")
            print(f"   - Industries: {len(self.industry_results)}")
            print(f"   - Total stocks: {total_stocks}")
            print(f"   - File size: {self.get_file_size(filename)}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None

    def get_file_size(self, filename: str) -> str:
        """Get human-readable file size"""
        try:
            import os
            size = os.path.getsize(filename)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

# Integration function
def get_industries_from_groups_dashboard():
    """Get all industries from the groups dashboard"""
    try:
        from groups_dashboard import GroupsDashboard
        
        print("🔄 Getting all industries from groups dashboard...")
        dashboard = GroupsDashboard()
        
        # Analyze Industry groups
        industry_data = dashboard.analyze_group_type('Industry', use_alternative=True)
        
        if industry_data is not None and 'Name' in industry_data.columns:
            industries = industry_data['Name'].tolist()
            print(f"✅ Retrieved {len(industries)} industries")
            return industries
        else:
            print("❌ No industry data available")
            return []
            
    except Exception as e:
        print(f"❌ Error getting industries: {e}")
        return []

# Main execution function
def run_full_industry_screener():
    """Run the complete full-scale industry screening"""
    print("🚀 STARTING FULL-SCALE INDUSTRY SCREENER")
    print("="*60)
    
    # Initialize screener
    screener = IndustryStockScreener()
    
    # Get all industries
    industries_list = get_industries_from_groups_dashboard()
    
    if not industries_list:
        print("❌ No industries found")
        return None
    
    print(f"📊 Found {len(industries_list)} industries to process")
    
    # Process all industries
    results = screener.process_all_industries(industries_list, batch_size=3)
    
    if results:
        # Display comprehensive results
        screener.display_summary_results(top_per_industry=5)
        
        # Export everything
        export_file = screener.export_comprehensive_results()
        
        print(f"\n🎉 FULL INDUSTRY SCREENING COMPLETE!")
        print(f"📁 Results saved to: {export_file}")
        
        return screener
    else:
        print("❌ No results obtained")
        return None

# Example usage
if __name__ == "__main__":
    # Run the full screener
    screener = run_full_industry_screener()
    
    if screener:
        print("\n🔧 SCREENER OBJECT AVAILABLE FOR FURTHER ANALYSIS:")
        print("=" * 50)
        print("# Access specific industry results:")
        print("uranium_stocks = screener.industry_results.get('Uranium')")
        print("\n# Get all industries with results:")
        print("available_industries = list(screener.industry_results.keys())")
        print("\n# Re-export with different filename:")
        print("screener.export_comprehensive_results('my_custom_results.xlsx')")