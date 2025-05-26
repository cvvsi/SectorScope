"""
Production Stock Scanner - Volume Pre-Filtered
Uses FinViz Technical screener for volume pre-filtering, then applies detailed analysis
Focuses on dollar volume (price × volume) as the primary liquidity filter
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from finvizfinance.screener.technical import Technical
from finvizfinance.quote import finvizfinance
import warnings
warnings.filterwarnings('ignore')

class ProductionStockScanner:
    def __init__(self, account_size=1000000):
        self.technical_screener = Technical()
        self.account_size = account_size
        self.vol_minimum = max(account_size * 5, 3000000)  # Dollar volume minimum
        self.adr_minimum = 3.5
        self.big_move_percent = 25
        self.atr_multiplier_20 = 1
        self.atr_multiplier_50 = 1
        
        # Connection settings
        self.max_retries = 3
        self.base_delay = 1
        
        print(f"🎯 Production Scanner initialized:")
        print(f"   Account size: ${account_size:,}")
        print(f"   Dollar volume minimum: ${self.vol_minimum:,}")
        print(f"   ADR minimum: {self.adr_minimum}%")
        print(f"   Big move minimum: {self.big_move_percent}%")

    def stage1_volume_prefilter(self):
        """
        Stage 1: Pre-filter by volume using Technical screener
        Focus on high-volume stocks to get better dollar volume candidates
        """
        print("\n🔍 Stage 1: Volume pre-filtering with Technical screener...")
        
        try:
            # No specific filters - let FinViz return volume-sorted data
            self.technical_screener.set_filter(filters_dict={})
            
            # Get candidates sorted by volume (highest first)
            self.candidates = self.technical_screener.screener_view(
                order='Average Volume (3 Month)',
                limit=1500,  # Get top 1500 by volume
                ascend=False,  # Highest volume first
                verbose=1,
                sleep_sec=1
            )
            
            if self.candidates is not None and not self.candidates.empty:
                print(f"✅ Retrieved {len(self.candidates)} volume-sorted candidates")
                
                # Clean up the data
                df = self.candidates.copy()
                
                # Remove funds/ETFs
                if 'Company' in df.columns:
                    fund_patterns = ['ETF', 'FUND', 'TRUST', 'REIT', 'INDEX', 'SPDR', 'ISHARES']
                    is_fund_mask = df['Company'].str.upper().str.contains('|'.join(fund_patterns), na=False)
                    df = df[~is_fund_mask]
                    print(f"📊 After removing funds/ETFs: {len(df)} candidates")
                
                # Convert volume and price to numeric for dollar volume calculation
                if 'Average Volume (3 Month)' in df.columns and 'Price' in df.columns:
                    # Clean volume data (remove M, K suffixes)
                    df['Volume_Clean'] = df['Average Volume (3 Month)'].astype(str).str.replace(',', '')
                    df['Volume_Clean'] = df['Volume_Clean'].str.replace('M', '').astype(float) * 1000000
                    df['Volume_Clean'] = df['Volume_Clean'].where(
                        ~df['Average Volume (3 Month)'].astype(str).str.contains('K'), 
                        df['Volume_Clean'].str.replace('K', '').astype(float) * 1000
                    )
                    
                    # Clean price data
                    df['Price_Clean'] = pd.to_numeric(df['Price'], errors='coerce')
                    
                    # Calculate dollar volume
                    df['Dollar_Volume'] = df['Volume_Clean'] * df['Price_Clean']
                    
                    # Filter by dollar volume threshold
                    df = df[df['Dollar_Volume'] >= self.vol_minimum]
                    print(f"📊 After dollar volume filter (>${self.vol_minimum:,}): {len(df)} candidates")
                    
                    # Sort by dollar volume
                    df = df.sort_values('Dollar_Volume', ascending=False)
                
                self.candidates = df
                return df
            else:
                print("❌ No candidates returned from screener")
                return None
                
        except Exception as e:
            print(f"❌ Stage 1 error: {e}")
            return None

    def stage2_detailed_analysis(self, max_analyze=300):
        """
        Stage 2: Get detailed data for top dollar volume candidates
        """
        if self.candidates is None or self.candidates.empty:
            print("❌ No candidates for detailed analysis")
            return None
            
        print(f"\n🔬 Stage 2: Detailed analysis of top {min(len(self.candidates), max_analyze)} dollar volume stocks...")
        
        # Take top candidates by dollar volume
        top_candidates = self.candidates.head(max_analyze)
        tickers = top_candidates['Ticker'].tolist()
        
        print(f"📊 Analyzing {len(tickers)} high dollar volume tickers...")
        
        self.detailed_data = []
        successful = 0
        failed = 0
        
        for i, ticker in enumerate(tickers):
            try:
                if i % 25 == 0:  # Progress update every 25 stocks
                    print(f"📈 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%) - Last: {ticker}")
                
                # Get detailed fundamental data
                stock = finvizfinance(ticker)
                fundament = stock.ticker_fundament()
                
                if fundament and isinstance(fundament, dict):
                    fundament['Ticker'] = ticker
                    self.detailed_data.append(fundament)
                    successful += 1
                else:
                    failed += 1
                
                # Rate limiting - be gentle with the API
                time.sleep(0.5)
                
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"⚠️ Failed {ticker}: {str(e)[:50]}")
                continue
        
        print(f"✅ Successfully analyzed: {successful} stocks")
        print(f"❌ Failed: {failed} stocks")
        
        if self.detailed_data:
            self.detailed_df = pd.DataFrame(self.detailed_data)
            print(f"📊 Created detailed dataset with {len(self.detailed_df)} stocks")
            return self.detailed_df
        else:
            print("❌ No detailed data collected")
            return None

    def stage3_apply_scan_criteria(self):
        """
        Stage 3: Apply your exact scan criteria to the detailed dataset
        """
        if not hasattr(self, 'detailed_df') or self.detailed_df is None:
            print("❌ No detailed data for scan criteria")
            return None
            
        print(f"\n🎯 Stage 3: Applying scan criteria to {len(self.detailed_df)} stocks...")
        
        df = self.detailed_df.copy()
        
        # Convert string percentages and numbers to numeric
        numeric_columns = [
            'Price', 'Avg Volume', 'ATR (14)', 'SMA20', 'SMA50', 'SMA200',
            'Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year',
            'Volatility W', 'Volatility M', 'Volume'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.replace(',', '')
                    .str.replace('B', '').str.replace('M', '').str.replace('K', ''),
                    errors='coerce'
                )
        
        print("🔄 Calculating scan conditions...")
        
        # Condition 1: ADR (Average Daily Range) > 3.5%
        # Use ATR as percentage of price
        if 'ATR (14)' in df.columns and 'Price' in df.columns:
            df['ADR_Percent'] = (df['ATR (14)'] / df['Price']) * 100
            df['Condition1_ADR'] = df['ADR_Percent'] > self.adr_minimum
        else:
            # Fallback to weekly volatility
            df['ADR_Percent'] = df.get('Volatility W', 0)
            df['Condition1_ADR'] = df['ADR_Percent'] > self.adr_minimum
        
        # Condition 2: Dollar Volume > minimum threshold
        if 'Avg Volume' in df.columns and 'Price' in df.columns:
            # Convert volume - FinViz shows in different units
            df['Volume_Numeric'] = df['Avg Volume'].copy()
            
            # Handle M (millions) and K (thousands) in volume
            mask_m = df['Avg Volume'].astype(str).str.contains('M', na=False)
            mask_k = df['Avg Volume'].astype(str).str.contains('K', na=False)
            
            if mask_m.any():
                df.loc[mask_m, 'Volume_Numeric'] = pd.to_numeric(
                    df.loc[mask_m, 'Avg Volume'].astype(str).str.replace('M', ''), errors='coerce'
                ) * 1000000
            
            if mask_k.any():
                df.loc[mask_k, 'Volume_Numeric'] = pd.to_numeric(
                    df.loc[mask_k, 'Avg Volume'].astype(str).str.replace('K', ''), errors='coerce'
                ) * 1000
            
            df['Dollar_Volume'] = df['Volume_Numeric'] * df['Price']
            df['Condition2_Volume'] = df['Dollar_Volume'] >= self.vol_minimum
        else:
            df['Condition2_Volume'] = False
        
        # Condition 3: Big Move over 30 days (25%+)
        # Use quarterly performance as proxy for 30-day big moves
        if 'Perf Quarter' in df.columns:
            df['Condition3_BigMove'] = abs(df['Perf Quarter']) >= self.big_move_percent
        elif 'Perf Month' in df.columns:
            # Use monthly * 1.3 as approximation for quarterly
            df['Condition3_BigMove'] = abs(df['Perf Month'] * 1.3) >= self.big_move_percent
        else:
            df['Condition3_BigMove'] = False
        
        # Condition 4: Price > (SMA20 - ATR)
        if all(col in df.columns for col in ['Price', 'SMA20', 'ATR (14)']):
            # SMA20 in FinViz is percentage distance from current price
            # Convert to actual SMA price: Current Price / (1 + SMA20_percent/100)
            df['SMA20_Price'] = df['Price'] / (1 + df['SMA20']/100)
            df['ATR_Buffer_20'] = self.atr_multiplier_20 * df['ATR (14)']
            df['SMA20_Threshold'] = df['SMA20_Price'] - df['ATR_Buffer_20']
            df['Condition4_SMA20'] = df['Price'] > df['SMA20_Threshold']
        else:
            # Fallback: price within reasonable distance of SMA20
            df['Condition4_SMA20'] = df.get('SMA20', 0) > -10  # Within 10% of SMA20
        
        # Condition 5: Price > (SMA50 - ATR)
        if all(col in df.columns for col in ['Price', 'SMA50', 'ATR (14)']):
            df['SMA50_Price'] = df['Price'] / (1 + df['SMA50']/100)
            df['ATR_Buffer_50'] = self.atr_multiplier_50 * df['ATR (14)']
            df['SMA50_Threshold'] = df['SMA50_Price'] - df['ATR_Buffer_50']
            df['Condition5_SMA50'] = df['Price'] > df['SMA50_Threshold']
        else:
            # Fallback: price within reasonable distance of SMA50
            df['Condition5_SMA50'] = df.get('SMA50', 0) > -15  # Within 15% of SMA50
        
        # Condition 6: Trend confirmation (current > 50 days ago)
        # Use monthly performance as proxy
        if 'Perf Month' in df.columns:
            df['Condition6_Trend'] = df['Perf Month'] > -5  # Allow small negative but not major decline
        else:
            df['Condition6_Trend'] = True
        
        # Calculate total conditions met
        condition_columns = [col for col in df.columns if col.startswith('Condition')]
        df['Total_Conditions_Met'] = df[condition_columns].sum(axis=1)
        df['Passes_All_Conditions'] = df['Total_Conditions_Met'] == len(condition_columns)
        
        # Summary statistics
        print(f"📊 Scan Results Summary:")
        for i, col in enumerate(condition_columns, 1):
            count = df[col].sum()
            print(f"   Condition {i} ({col.split('_')[1]}): {count} stocks")
        
        total_pass = df['Passes_All_Conditions'].sum()
        print(f"✅ FINAL RESULTS: {total_pass} stocks pass ALL conditions")
        
        # Filter to final results
        self.final_results = df[df['Passes_All_Conditions']].copy()
        self.final_results = self.final_results.sort_values('Dollar_Volume', ascending=False)
        
        return self.final_results

    def display_results(self, n=25):
        """
        Display top scan results
        """
        if self.final_results is None or self.final_results.empty:
            print("❌ No final results to display")
            return
            
        print(f"\n🏆 TOP {min(n, len(self.final_results))} SCAN RESULTS:")
        print("="*100)
        
        display_cols = [
            'Ticker', 'Company', 'Sector', 'Price', 'Dollar_Volume', 
            'ADR_Percent', 'Perf Month', 'Perf Quarter', 'Total_Conditions_Met'
        ]
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in self.final_results.columns]
        
        top_results = self.final_results.head(n)
        
        # Format dollar volume for readability
        if 'Dollar_Volume' in top_results.columns:
            top_results = top_results.copy()
            top_results['Dollar_Volume_M'] = (top_results['Dollar_Volume'] / 1000000).round(1)
            available_cols = [col.replace('Dollar_Volume', 'Dollar_Volume_M') for col in available_cols]
        
        print(top_results[available_cols].to_string(index=False, float_format='%.2f'))

    def export_results(self, filename=None):
        """
        Export comprehensive results to CSV
        """
        if self.final_results is None or self.final_results.empty:
            print("❌ No results to export")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_scan_results_{timestamp}.csv"
        
        # Prepare export data
        export_df = self.final_results.copy()
        
        # Add readable dollar volume
        export_df['Dollar_Volume_Millions'] = (export_df['Dollar_Volume'] / 1000000).round(2)
        
        # Select key columns for export
        key_columns = [
            'Ticker', 'Company', 'Sector', 'Industry', 'Price', 'Dollar_Volume_Millions',
            'ADR_Percent', 'ATR (14)', 'SMA20', 'SMA50', 'SMA200',
            'Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year',
            'Total_Conditions_Met'
        ]
        
        # Add all condition columns
        condition_cols = [col for col in export_df.columns if 'Condition' in col]
        key_columns.extend(condition_cols)
        
        # Filter to available columns
        available_columns = [col for col in key_columns if col in export_df.columns]
        
        final_export = export_df[available_columns].copy()
        final_export.to_csv(filename, index=False)
        
        print(f"✅ Results exported to: {filename}")
        print(f"📊 Exported {len(final_export)} stocks with detailed analysis")
        
        return filename

    def save_cache(self, filename='production_scan_cache.pkl'):
        """
        Save detailed data for faster future runs
        """
        if hasattr(self, 'detailed_df') and self.detailed_df is not None:
            cache_data = {
                'detailed_df': self.detailed_df,
                'timestamp': datetime.now(),
                'account_size': self.account_size,
                'vol_minimum': self.vol_minimum
            }
            
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"💾 Cached {len(self.detailed_df)} stocks to {filename}")

    def load_cache(self, filename='production_scan_cache.pkl', max_age_hours=6):
        """
        Load cached data if recent enough
        """
        try:
            if os.path.exists(filename):
                import pickle
                with open(filename, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check cache age
                cache_age = (datetime.now() - cache_data['timestamp']).total_seconds() / 3600
                
                if cache_age <= max_age_hours:
                    self.detailed_df = cache_data['detailed_df']
                    print(f"📂 Loaded {len(self.detailed_df)} stocks from cache")
                    print(f"   Cache age: {cache_age:.1f} hours")
                    return True
                else:
                    print(f"⏰ Cache too old ({cache_age:.1f} hours), will refresh")
                    return False
            else:
                print("📂 No cache file found")
                return False
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False

def run_production_scan(account_size=1000000, use_cache=True, max_analyze=300):
    """
    Run the complete production scan
    """
    print("🚀 PRODUCTION STOCK SCANNER")
    print("="*60)
    print("HIGH DOLLAR VOLUME → DETAILED ANALYSIS → SCAN CRITERIA")
    print("="*60)
    
    # Initialize scanner
    scanner = ProductionStockScanner(account_size=account_size)
    
    # Try to load cache first
    cache_loaded = False
    if use_cache:
        cache_loaded = scanner.load_cache()
    
    if not cache_loaded:
        # Stage 1: Volume pre-filtering
        candidates = scanner.stage1_volume_prefilter()
        if candidates is None or candidates.empty:
            print("❌ No high-volume candidates found")
            return None
        
        # Stage 2: Detailed analysis of top dollar volume stocks
        detailed_data = scanner.stage2_detailed_analysis(max_analyze=max_analyze)
        if detailed_data is None:
            print("❌ Failed to get detailed data")
            return None
        
        # Save cache for next time
        scanner.save_cache()
    
    # Stage 3: Apply scan criteria
    results = scanner.stage3_apply_scan_criteria()
    if results is None:
        print("❌ No stocks meet all criteria")
        return scanner  # Return scanner even if no results
    
    # Display and export results
    scanner.display_results(25)
    filename = scanner.export_results()
    
    print(f"\n🎯 SCAN COMPLETE!")
    print(f"📈 Found {len(results)} stocks meeting ALL scan criteria")
    print(f"📄 Detailed results saved to: {filename}")
    
    return scanner

if __name__ == "__main__":
    # Run the production scanner
    scanner = run_production_scan(
        account_size=1000000,  # Adjust to your account size
        use_cache=True,        # Use cache for faster subsequent runs
        max_analyze=300        # Analyze top 300 dollar volume stocks
    )
    
    if scanner:
        print("\n🎯 Production scan complete!")
        print("📊 Scanner object available for further analysis")