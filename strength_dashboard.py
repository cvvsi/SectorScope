"""
Enhanced Stock Strength Dashboard with Sector/Industry Rotation
Combines Performance + Overview screeners for comprehensive analysis
Maximum sample size + CSV exports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from finvizfinance.screener.performance import Performance
from finvizfinance.screener.overview import Overview
import warnings
warnings.filterwarnings('ignore')

class EnhancedStrengthDashboard:
    def __init__(self):
        self.performance_screener = Performance()
        self.overview_screener = Overview()
        self.performance_data = None
        self.overview_data = None
        self.merged_data = None
        
        print("🚀 ENHANCED STOCK STRENGTH DASHBOARD - MAXIMUM COVERAGE")
        print("="*70)
        print("📊 Performance Data + Sector/Industry Analysis + CSV Export")
        print("="*70)

    def fetch_performance_data(self, limit=2000, max_retries=3):
        """Fetch performance data with retry logic - MAXIMUM LIMIT"""
        print(f"\n📈 Fetching performance data (limit: {limit})...")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                self.performance_screener.set_filter(filters_dict={})
                self.performance_data = self.performance_screener.screener_view(
                    order='Performance (Month)',  # Sort by monthly performance
                    limit=limit,
                    ascend=False,
                    verbose=1,
                    sleep_sec=1  # Faster for large datasets
                )
                
                if self.performance_data is not None and not self.performance_data.empty:
                    print(f"✅ Performance data: {len(self.performance_data)} stocks")
                    return self.performance_data
                else:
                    print(f"❌ Attempt {attempt + 1} failed - no data")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        print("❌ Failed to fetch performance data after all retries")
        return None

    def fetch_overview_data(self, limit=8000, max_retries=3):
        """Fetch overview data - MASSIVE LIMIT for maximum overlap"""
        print(f"\n📋 Fetching overview data (limit: {limit})...")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                self.overview_screener.set_filter(filters_dict={})
                self.overview_data = self.overview_screener.screener_view(
                    order='Change',  # Use performance-related order for better overlap
                    limit=limit,
                    ascend=False,  # Best performers first
                    verbose=1,
                    sleep_sec=1
                )
                
                if self.overview_data is not None and not self.overview_data.empty:
                    print(f"✅ Overview data: {len(self.overview_data)} stocks")
                    return self.overview_data
                else:
                    print(f"❌ Attempt {attempt + 1} failed - no data")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        print("❌ Failed to fetch overview data after all retries")
        return None

    def merge_datasets(self):
        """Merge performance and overview data"""
        if self.performance_data is None or self.overview_data is None:
            print("❌ Cannot merge - missing data")
            return None
            
        print("\n🔄 Merging performance and overview data...")
        
        # Clean and prepare data
        perf_df = self.performance_data.copy()
        overview_df = self.overview_data.copy()
        
        print(f"📊 Before cleaning - Performance: {len(perf_df)}, Overview: {len(overview_df)}")
        
        # Remove funds/ETFs from both datasets
        fund_patterns = ['ETF', 'FUND', 'TRUST', 'REIT', 'INDEX', 'SPDR', 'ISHARES']
        
        if 'Company' in perf_df.columns:
            is_fund_mask = perf_df['Company'].str.upper().str.contains('|'.join(fund_patterns), na=False)
            perf_df = perf_df[~is_fund_mask]
        
        if 'Company' in overview_df.columns:
            is_fund_mask = overview_df['Company'].str.upper().str.contains('|'.join(fund_patterns), na=False)
            overview_df = overview_df[~is_fund_mask]
        
        print(f"📊 After fund removal - Performance: {len(perf_df)}, Overview: {len(overview_df)}")
        
        # Merge on Ticker - keep only overlapping stocks
        overview_merge_cols = ['Ticker']
        if 'Company' in overview_df.columns:
            overview_merge_cols.append('Company')
        if 'Sector' in overview_df.columns:
            overview_merge_cols.append('Sector')
        if 'Industry' in overview_df.columns:
            overview_merge_cols.append('Industry')
        # Add other useful columns if they exist
        for col in ['Market Cap', 'P/E', 'Price', 'Country']:
            if col in overview_df.columns:
                overview_merge_cols.append(col)
        
        self.merged_data = pd.merge(
            perf_df, 
            overview_df[overview_merge_cols], 
            on='Ticker', 
            how='inner',
            suffixes=('_perf', '_overview')
        )
        
        print(f"✅ Merged dataset: {len(self.merged_data)} stocks")
        
        # Check sector/industry coverage
        if 'Sector' in self.merged_data.columns:
            sector_counts = self.merged_data['Sector'].value_counts()
            print(f"📊 Available sectors: {len(sector_counts)}")
            print(f"📊 Top sectors: {list(sector_counts.head(5).index)}")
            print(f"📊 Sector distribution: {dict(sector_counts.head(8))}")
        
        if 'Industry' in self.merged_data.columns:
            print(f"📊 Available industries: {self.merged_data['Industry'].nunique()}")
        
        return self.merged_data

    def process_strength_metrics(self):
        """Calculate strength metrics and rankings"""
        if self.merged_data is None:
            print("❌ No merged data to process")
            return None
            
        print("\n📊 Calculating strength metrics...")
        
        df = self.merged_data.copy()
        
        # Performance columns
        perf_columns = ['Perf Week', 'Perf Month', 'Perf Quart', 'Perf Half', 'Perf Year', 'Perf YTD']
        
        # Convert to numeric
        for col in perf_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.replace(',', ''), 
                    errors='coerce'
                )
        
        # Individual timeframe rankings (percentile ranks)
        for col in perf_columns:
            if col in df.columns:
                df[f'{col}_Rank'] = df[col].rank(pct=True) * 100
        
        # Composite strength scores
        short_term_cols = [col for col in ['Perf Week', 'Perf Month'] if col in df.columns]
        if short_term_cols:
            df['Short_Term_Strength'] = df[short_term_cols].mean(axis=1, skipna=True)
            df['ST_Rank'] = df['Short_Term_Strength'].rank(pct=True) * 100
        
        long_term_cols = [col for col in ['Perf Quart', 'Perf Half', 'Perf Year'] if col in df.columns]
        if long_term_cols:
            df['Long_Term_Strength'] = df[long_term_cols].mean(axis=1, skipna=True)
            df['LT_Rank'] = df['Long_Term_Strength'].rank(pct=True) * 100
        
        # Overall strength
        all_perf_cols = [col for col in perf_columns if col in df.columns]
        if all_perf_cols:
            df['Overall_Strength'] = df[all_perf_cols].mean(axis=1, skipna=True)
            df['Overall_Rank'] = df['Overall_Strength'].rank(pct=True) * 100
        
        # Consistency score
        positive_count = (df[all_perf_cols] > 0).sum(axis=1)
        total_timeframes = df[all_perf_cols].notna().sum(axis=1)
        df['Consistency'] = (positive_count / total_timeframes * 100).round(1)
        
        self.processed_data = df
        print(f"✅ Processed {len(df)} stocks with strength metrics")
        
        return df

    def analyze_sector_rotation(self, min_stocks=5):
        """Analyze sector performance like ETF dashboard"""
        if self.processed_data is None:
            print("❌ No processed data for sector analysis")
            return None
            
        print("\n🔄 Analyzing sector rotation...")
        
        df = self.processed_data.copy()
        
        # Check if we have sector data
        if 'Sector' not in df.columns:
            print("❌ No sector data available")
            return None
        
        # Sector analysis across timeframes
        perf_columns = ['Perf Week', 'Perf Month', 'Perf Quart', 'Perf Half', 'Perf Year', 'Perf YTD']
        available_perf_cols = [col for col in perf_columns if col in df.columns]
        
        agg_dict = {col: 'mean' for col in available_perf_cols}
        agg_dict.update({
            'Overall_Strength': 'mean',
            'Consistency': 'mean',
            'Ticker': 'count'
        })
        
        sector_analysis = df.groupby('Sector').agg(agg_dict).round(2)
        
        # Filter sectors with minimum stocks
        sector_analysis = sector_analysis[sector_analysis['Ticker'] >= min_stocks]
        
        # Add sector rankings
        for col in available_perf_cols:
            if col in sector_analysis.columns:
                sector_analysis[f'{col}_Sector_Rank'] = sector_analysis[col].rank(ascending=False)
        
        if 'Overall_Strength' in sector_analysis.columns:
            sector_analysis['Overall_Sector_Rank'] = sector_analysis['Overall_Strength'].rank(ascending=False)
            # Sort by overall strength
            sector_analysis = sector_analysis.sort_values('Overall_Strength', ascending=False)
        
        self.sector_analysis = sector_analysis
        return sector_analysis

    def analyze_industry_rotation(self, min_stocks=3):
        """Analyze industry performance"""
        if self.processed_data is None:
            print("❌ No processed data for industry analysis")
            return None
            
        print("\n🔄 Analyzing industry rotation...")
        
        df = self.processed_data.copy()
        
        # Check if we have industry data
        if 'Industry' not in df.columns:
            print("❌ No industry data available")
            return None
        
        # Industry analysis
        perf_columns = ['Perf Week', 'Perf Month', 'Perf Quart', 'Perf Half', 'Perf Year', 'Perf YTD']
        available_perf_cols = [col for col in perf_columns if col in df.columns]
        
        agg_dict = {col: 'mean' for col in available_perf_cols}
        agg_dict.update({
            'Overall_Strength': 'mean',
            'Consistency': 'mean',
            'Ticker': 'count'
        })
        
        industry_analysis = df.groupby('Industry').agg(agg_dict).round(2)
        
        # Filter industries with minimum stocks
        industry_analysis = industry_analysis[industry_analysis['Ticker'] >= min_stocks]
        
        # Sort by overall strength
        if 'Overall_Strength' in industry_analysis.columns:
            industry_analysis = industry_analysis.sort_values('Overall_Strength', ascending=False)
        
        self.industry_analysis = industry_analysis
        return industry_analysis

    def export_to_csv(self, timestamp=None):
        """Export all datasets to CSV files"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n💾 Exporting data to CSV files...")
        
        exported_files = []
        
        # Export main processed data
        if self.processed_data is not None:
            filename = f"strength_dashboard_stocks_{timestamp}.csv"
            self.processed_data.to_csv(filename, index=False)
            exported_files.append(filename)
            print(f"✅ Exported: {filename} ({len(self.processed_data)} stocks)")
        
        # Export sector analysis
        if hasattr(self, 'sector_analysis') and self.sector_analysis is not None:
            filename = f"strength_dashboard_sectors_{timestamp}.csv"
            self.sector_analysis.to_csv(filename, index=True)
            exported_files.append(filename)
            print(f"✅ Exported: {filename} ({len(self.sector_analysis)} sectors)")
        
        # Export industry analysis
        if hasattr(self, 'industry_analysis') and self.industry_analysis is not None:
            filename = f"strength_dashboard_industries_{timestamp}.csv"
            self.industry_analysis.to_csv(filename, index=True)
            exported_files.append(filename)
            print(f"✅ Exported: {filename} ({len(self.industry_analysis)} industries)")
        
        # Export top performers by sector
        if hasattr(self, 'sector_analysis') and self.processed_data is not None:
            sector_leaders_data = []
            top_sectors = self.sector_analysis.head(10).index
            
            for sector in top_sectors:
                sector_stocks = self.get_sector_leaders(sector, 20)  # Top 20 per sector
                if sector_stocks is not None and not sector_stocks.empty:
                    sector_stocks = sector_stocks.copy()
                    sector_stocks['Sector_Rank'] = sector_stocks['Overall_Strength'].rank(ascending=False)
                    sector_leaders_data.append(sector_stocks)
            
            if sector_leaders_data:
                sector_leaders_df = pd.concat(sector_leaders_data, ignore_index=True)
                filename = f"strength_dashboard_sector_leaders_{timestamp}.csv"
                sector_leaders_df.to_csv(filename, index=False)
                exported_files.append(filename)
                print(f"✅ Exported: {filename} ({len(sector_leaders_df)} sector leaders)")
        
        # Export raw performance data
        if self.performance_data is not None:
            filename = f"raw_performance_data_{timestamp}.csv"
            self.performance_data.to_csv(filename, index=False)
            exported_files.append(filename)
            print(f"✅ Exported: {filename} ({len(self.performance_data)} raw performance stocks)")
        
        print(f"\n📁 All exports complete! Files created:")
        for file in exported_files:
            print(f"   📄 {file}")
        
        return exported_files

    def get_sector_leaders(self, sector, n=10):
        """Get top stocks within a sector"""
        if self.processed_data is None:
            return None
            
        sector_stocks = self.processed_data[
            self.processed_data['Sector'].str.contains(sector, case=False, na=False)
        ]
        
        if 'Overall_Strength' in sector_stocks.columns:
            return sector_stocks.sort_values('Overall_Strength', ascending=False).head(n)
        else:
            return sector_stocks.head(n)

    def get_industry_leaders(self, industry, n=10):
        """Get top stocks within an industry"""
        if self.processed_data is None:
            return None
            
        industry_stocks = self.processed_data[
            self.processed_data['Industry'].str.contains(industry, case=False, na=False)
        ]
        
        if 'Overall_Strength' in industry_stocks.columns:
            return industry_stocks.sort_values('Overall_Strength', ascending=False).head(n)
        else:
            return industry_stocks.head(n)

    def display_sector_rotation_summary(self):
        """Display sector rotation analysis like your ETF dashboard"""
        if self.sector_analysis is None:
            print("❌ No sector analysis available")
            return
            
        print("\n🏢 SECTOR ROTATION ANALYSIS")
        print("="*80)
        print("📊 Timeframe Performance by Sector (% Average)")
        print("="*80)
        
        # Display sector performance across timeframes
        display_cols = []
        if 'Perf Week' in self.sector_analysis.columns:
            display_cols.append('Perf Week')
        if 'Perf Month' in self.sector_analysis.columns:
            display_cols.append('Perf Month')
        if 'Perf Quart' in self.sector_analysis.columns:
            display_cols.append('Perf Quart')
        if 'Perf Half' in self.sector_analysis.columns:
            display_cols.append('Perf Half')
        if 'Perf Year' in self.sector_analysis.columns:
            display_cols.append('Perf Year')
        if 'Perf YTD' in self.sector_analysis.columns:
            display_cols.append('Perf YTD')
        
        if 'Overall_Strength' in self.sector_analysis.columns:
            display_cols.append('Overall_Strength')
        if 'Consistency' in self.sector_analysis.columns:
            display_cols.append('Consistency')
        if 'Ticker' in self.sector_analysis.columns:
            display_cols.append('Ticker')
        
        if display_cols:
            sector_display = self.sector_analysis[display_cols].copy()
            sector_display.columns = [col.replace('Perf ', '') for col in sector_display.columns]
            
            print(sector_display.to_string())
            
            # Show leading and lagging sectors
            if 'Overall_Strength' in self.sector_analysis.columns:
                print(f"\n🏆 TOP PERFORMING SECTORS:")
                print("-" * 70)
                top_sectors = self.sector_analysis.head(8)
                for idx, (sector, data) in enumerate(top_sectors.iterrows(), 1):
                    overall = data.get('Overall_Strength', 0)
                    ticker_count = int(data.get('Ticker', 0))
                    consistency = data.get('Consistency', 0)
                    print(f"{idx}. {sector[:35]:35} | Overall: {overall:+6.2f}% | Stocks: {ticker_count:3} | Consistency: {consistency:5.1f}%")

    def display_industry_rotation_summary(self):
        """Display top/bottom industries"""
        if self.industry_analysis is None:
            print("❌ No industry analysis available")
            return
            
        print("\n🏭 INDUSTRY ROTATION ANALYSIS")
        print("="*80)
        
        if 'Overall_Strength' in self.industry_analysis.columns:
            print(f"🏆 TOP 15 PERFORMING INDUSTRIES:")
            print("-" * 80)
            top_industries = self.industry_analysis.head(15)
            for idx, (industry, data) in enumerate(top_industries.iterrows(), 1):
                overall = data.get('Overall_Strength', 0)
                ticker_count = int(data.get('Ticker', 0))
                consistency = data.get('Consistency', 0)
                print(f"{idx:2}. {industry[:45]:45} | {overall:+6.2f}% | {ticker_count:2} stocks | {consistency:5.1f}%")

    def display_stock_leaders_by_sector(self, top_n_sectors=6, stocks_per_sector=8):
        """Display top stocks within top sectors"""
        if self.sector_analysis is None or self.processed_data is None:
            print("❌ No data available for sector leaders")
            return
            
        print(f"\n🌟 TOP STOCKS IN LEADING SECTORS")
        print("="*90)
        
        top_sectors = self.sector_analysis.head(top_n_sectors).index
        
        for sector in top_sectors:
            print(f"\n🏢 {sector.upper()}")
            print("-" * 70)
            
            sector_leaders = self.get_sector_leaders(sector, stocks_per_sector)
            if sector_leaders is not None and not sector_leaders.empty:
                for idx, (_, stock) in enumerate(sector_leaders.iterrows(), 1):
                    ticker = stock['Ticker']
                    company = stock.get('Company', 'N/A')
                    if len(str(company)) > 30:
                        company = str(company)[:30]
                    overall = stock.get('Overall_Strength', 0)
                    consistency = stock.get('Consistency', 0)
                    month_perf = stock.get('Perf Month', 0)
                    print(f"{idx}. {ticker:6} | {company:30} | Overall: {overall:+6.2f}% | Month: {month_perf:+6.2f}% | Consistency: {consistency:5.1f}%")

    def run_full_analysis(self, performance_limit=2000, overview_limit=8000):
        """Run complete enhanced analysis with MAXIMUM coverage"""
        
        # Step 1: Fetch data
        perf_data = self.fetch_performance_data(performance_limit)
        if perf_data is None:
            return None
            
        overview_data = self.fetch_overview_data(overview_limit)
        if overview_data is None:
            return None
        
        # Step 2: Merge datasets
        merged = self.merge_datasets()
        if merged is None:
            return None
        
        # Step 3: Process strength metrics
        processed = self.process_strength_metrics()
        if processed is None:
            return None
        
        # Step 4: Analyze rotations
        sector_analysis = self.analyze_sector_rotation()
        industry_analysis = self.analyze_industry_rotation()
        
        # Step 5: Export to CSV
        exported_files = self.export_to_csv()
        
        # Step 6: Display comprehensive summary
        self.display_sector_rotation_summary()
        self.display_industry_rotation_summary()
        self.display_stock_leaders_by_sector()
        
        print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
        print(f"📊 Total stocks analyzed: {len(self.processed_data)}")
        if 'Sector' in self.processed_data.columns:
            print(f"🏢 Sectors covered: {self.processed_data['Sector'].nunique()}")
        if 'Industry' in self.processed_data.columns:
            print(f"🏭 Industries covered: {self.processed_data['Industry'].nunique()}")
        
        print(f"\n📁 CSV files exported: {len(exported_files)} files")
        
        return self

# Main execution
if __name__ == "__main__":
    dashboard = EnhancedStrengthDashboard()
    result = dashboard.run_full_analysis()
    
    # Additional interactive functions available:
    if result:
        print("\n🔧 INTERACTIVE FUNCTIONS:")
        print("dashboard.get_sector_leaders('Technology', 20)")
        print("dashboard.get_industry_leaders('Software', 15)")
        print("dashboard.sector_analysis")
        print("dashboard.industry_analysis")
        print("dashboard.processed_data")
        print("dashboard.export_to_csv()  # Re-export anytime")