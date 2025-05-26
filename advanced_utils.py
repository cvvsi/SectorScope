"""
Advanced FinViz Analysis Utilities
Additional functions for deeper market analysis and tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedAnalysis:
    def __init__(self, tracker):
        self.tracker = tracker
        self.data = tracker.processed_data if tracker.processed_data is not None else None
    
    def create_heatmap(self, focus='sector', top_n=15):
        """
        Create a performance heatmap by sector or industry
        """
        if self.data is None:
            print("❌ No data available")
            return
        
        # Get performance columns
        perf_cols = [col for col in self.data.columns if 'Perf' in col and col != 'Overall_Score']
        
        if not perf_cols:
            print("❌ No performance columns found")
            return
        
        # Group by focus area
        if focus.lower() == 'sector':
            grouped = self.data.groupby('Sector')[perf_cols].mean()
        else:
            grouped = self.data.groupby('Industry')[perf_cols].mean()
        
        # Get top N by overall performance
        overall_perf = grouped.mean(axis=1).sort_values(ascending=False)
        top_groups = overall_perf.head(top_n).index
        
        # Create heatmap data
        heatmap_data = grouped.loc[top_groups]
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
        plt.title(f'Performance Heatmap - Top {top_n} {focus.title()}s')
        plt.tight_layout()
        plt.show()
    
    def momentum_analysis(self, min_score=5.0):
        """
        Find stocks with consistent momentum across timeframes
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        # Get performance columns
        perf_cols = [col for col in self.data.columns if 'Perf' in col and 'Score' not in col]
        
        if len(perf_cols) < 2:
            print("❌ Need at least 2 performance timeframes")
            return None
        
        # Find stocks that are positive across most timeframes
        df = self.data.copy()
        
        # Count positive performance periods
        df['Positive_Periods'] = (df[perf_cols] > 0).sum(axis=1)
        df['Total_Periods'] = df[perf_cols].notna().sum(axis=1)
        df['Momentum_Ratio'] = df['Positive_Periods'] / df['Total_Periods']
        
        # Filter for strong momentum and minimum performance
        momentum_stocks = df[
            (df['Momentum_Ratio'] >= 0.7) &  # Positive in 70%+ of periods
            (df['Overall_Score'] >= min_score)  # Minimum overall performance
        ].copy()
        
        momentum_stocks = momentum_stocks.sort_values('Overall_Score', ascending=False)
        
        print(f"🚀 Found {len(momentum_stocks)} stocks with strong momentum")
        return momentum_stocks
    
    def volatility_analysis(self):
        """
        Analyze volatility across different timeframes
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        # Get performance columns
        perf_cols = [col for col in self.data.columns if 'Perf' in col and 'Score' not in col]
        
        if len(perf_cols) < 3:
            print("❌ Need at least 3 performance timeframes")
            return None
        
        df = self.data.copy()
        
        # Calculate volatility (standard deviation across timeframes)
        df['Volatility'] = df[perf_cols].std(axis=1, skipna=True)
        
        # Risk-adjusted return (Overall Score / Volatility)
        df['Risk_Adjusted_Return'] = df['Overall_Score'] / (df['Volatility'] + 0.1)  # Add small value to avoid division by zero
        
        # Categorize stocks
        df['Risk_Category'] = pd.cut(df['Volatility'], 
                                   bins=[0, df['Volatility'].quantile(0.33), 
                                        df['Volatility'].quantile(0.67), float('inf')],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        return df.sort_values('Risk_Adjusted_Return', ascending=False)
    
    def sector_rotation_signals(self):
        """
        Identify potential sector rotation signals
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        # Get short-term vs long-term performance by sector
        sector_stats = self.data.groupby('Sector').agg({
            'Short_Term_Score': 'mean',
            'Long_Term_Score': 'mean',
            'Overall_Score': 'mean',
            'Ticker': 'count'
        }).round(2)
        
        # Filter sectors with enough stocks
        sector_stats = sector_stats[sector_stats['Ticker'] >= 3]
        
        # Calculate rotation signals
        sector_stats['ST_vs_LT_Ratio'] = sector_stats['Short_Term_Score'] / (sector_stats['Long_Term_Score'] + 0.1)
        sector_stats['Momentum_Direction'] = np.where(
            sector_stats['ST_vs_LT_Ratio'] > 1.2, 'Accelerating',
            np.where(sector_stats['ST_vs_LT_Ratio'] < 0.8, 'Decelerating', 'Stable')
        )
        
        # Sort by short-term performance
        sector_stats = sector_stats.sort_values('Short_Term_Score', ascending=False)
        
        print("🔄 SECTOR ROTATION ANALYSIS:")
        print("=" * 60)
        print("Accelerating: Short-term outperforming long-term (momentum building)")
        print("Decelerating: Short-term underperforming long-term (momentum fading)")
        print("Stable: Consistent performance across timeframes")
        print("=" * 60)
        
        return sector_stats
    
    def find_breakout_candidates(self, min_recent_performance=10.0):
        """
        Find stocks that might be breaking out
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        df = self.data.copy()
        
        # Look for stocks with strong recent performance
        short_term_cols = [col for col in df.columns 
                          if any(term in col.lower() for term in ['day', 'week', '1d', '2d', '5d'])]
        
        if not short_term_cols:
            print("❌ No short-term performance columns found")
            return None
        
        # Calculate recent average performance
        df['Recent_Performance'] = df[short_term_cols].mean(axis=1, skipna=True)
        
        # Find potential breakouts
        breakout_candidates = df[
            (df['Recent_Performance'] >= min_recent_performance) &
            (df['Overall_Score'] > 0)  # Positive overall trend
        ].copy()
        
        breakout_candidates = breakout_candidates.sort_values('Recent_Performance', ascending=False)
        
        print(f"🚀 Found {len(breakout_candidates)} potential breakout candidates")
        return breakout_candidates
    
    def generate_watchlist(self, criteria='balanced', n=25):
        """
        Generate a focused watchlist based on different criteria
        """
        if self.data is None:
            print("❌ No data available")
            return None
        
        df = self.data.copy()
        
        if criteria.lower() == 'momentum':
            # Focus on momentum stocks
            momentum_stocks = self.momentum_analysis(min_score=3.0)
            if momentum_stocks is not None:
                watchlist = momentum_stocks.head(n)
        
        elif criteria.lower() == 'breakout':
            # Focus on breakout candidates
            breakout_stocks = self.find_breakout_candidates(min_recent_performance=5.0)
            if breakout_stocks is not None:
                watchlist = breakout_stocks.head(n)
        
        elif criteria.lower() == 'risk_adjusted':
            # Focus on risk-adjusted returns
            risk_analysis = self.volatility_analysis()
            if risk_analysis is not None:
                watchlist = risk_analysis.head(n)
        
        else:  # balanced
            # Balanced approach - top overall performers with decent momentum
            watchlist = df[
                (df['Overall_Score'] >= df['Overall_Score'].quantile(0.8)) &
                (df['Overall_Score'] > 0)
            ].sort_values('Overall_Score', ascending=False).head(n)
        
        print(f"📋 {criteria.upper()} WATCHLIST - Top {n} Stocks:")
        print("=" * 60)
        
        # Display key columns
        display_cols = ['Ticker', 'Company', 'Sector', 'Overall_Score', 'Overall_Rank']
        if 'Recent_Performance' in watchlist.columns:
            display_cols.append('Recent_Performance')
        if 'Risk_Adjusted_Return' in watchlist.columns:
            display_cols.append('Risk_Adjusted_Return')
        
        print(watchlist[display_cols].to_string(index=False))
        
        return watchlist
    
    def export_analysis(self, filename_base='finviz_analysis'):
        """
        Export analysis results to Excel file
        """
        if self.data is None:
            print("❌ No data available")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data
            self.data.to_excel(writer, sheet_name='All_Stocks', index=False)
            
            # Top performers
            top_50 = self.tracker.get_top_performers(50, 'overall')
            if top_50 is not None:
                top_50.to_excel(writer, sheet_name='Top_50_Overall', index=False)
            
            # Sector analysis
            sector_analysis, industry_analysis = self.tracker.analyze_by_sector_industry()
            if sector_analysis is not None:
                sector_analysis.to_excel(writer, sheet_name='Sector_Analysis')
            if industry_analysis is not None:
                industry_analysis.to_excel(writer, sheet_name='Industry_Analysis')
            
            # Momentum analysis
            momentum_stocks = self.momentum_analysis(min_score=2.0)
            if momentum_stocks is not None:
                momentum_stocks.to_excel(writer, sheet_name='Momentum_Stocks', index=False)
            
            # Watchlists
            for criteria in ['balanced', 'momentum', 'breakout']:
                watchlist = self.generate_watchlist(criteria, 25)
                if watchlist is not None:
                    watchlist.to_excel(writer, sheet_name=f'Watchlist_{criteria.title()}', index=False)
        
        print(f"✅ Analysis exported to: {filename}")

# Usage example
def run_advanced_analysis(tracker):
    """
    Run advanced analysis on the performance tracker data
    """
    if tracker is None or tracker.processed_data is None:
        print("❌ No tracker data available")
        return None
    
    analyzer = AdvancedAnalysis(tracker)
    
    print("\n" + "="*80)
    print("🔬 ADVANCED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Momentum analysis
    print("\n🚀 MOMENTUM ANALYSIS:")
    momentum_stocks = analyzer.momentum_analysis(min_score=5.0)
    if momentum_stocks is not None:
        print(f"Top 10 Momentum Stocks:")
        cols = ['Ticker', 'Company', 'Sector', 'Overall_Score', 'Momentum_Ratio']
        print(momentum_stocks[cols].head(10).to_string(index=False))
    
    # Sector rotation
    print("\n🔄 SECTOR ROTATION SIGNALS:")
    rotation_signals = analyzer.sector_rotation_signals()
    if rotation_signals is not None:
        print(rotation_signals.to_string())
    
    # Generate watchlists
    print("\n📋 GENERATING WATCHLISTS:")
    for criteria in ['momentum', 'breakout', 'balanced']:
        watchlist = analyzer.generate_watchlist(criteria, 10)
        print(f"\n{criteria.upper()} WATCHLIST:")
        if watchlist is not None:
            cols = ['Ticker', 'Company', 'Sector', 'Overall_Score']
            print(watchlist[cols].head(10).to_string(index=False))
    
    return analyzer

if __name__ == "__main__":
    print("🔬 Advanced Analysis Utilities Ready!")
    print("Usage: analyzer = run_advanced_analysis(your_tracker_object)")
    print("\nAvailable functions:")
    print("- analyzer.momentum_analysis()")
    print("- analyzer.sector_rotation_signals()")
    print("- analyzer.find_breakout_candidates()")
    print("- analyzer.generate_watchlist('momentum')")
    print("- analyzer.export_analysis()")