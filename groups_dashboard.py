"""
Groups Dashboard for Trading - Leading Sectors & Industries Performance
Uses finvizfinance.group modules to get sector/industry-level data with:
Name, Market Cap, Performance (Week), Performance (Month), Performance (Quarter), 
Performance (Half Year), Performance (Year), Performance (Year To Date)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import finvizfinance group modules
try:
    from finvizfinance.group.overview import Overview as GroupOverview
    from finvizfinance.group.valuation import Valuation as GroupValuation
    from finvizfinance.group.performance import Performance as GroupPerformance
    GROUP_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Group modules not available: {e}")
    print("💡 You may need to update finvizfinance or check if these modules exist")
    GROUP_MODULES_AVAILABLE = False

class GroupsDashboard:
    def __init__(self):
        """Initialize the Groups Dashboard"""
        self.group_data = {}  # Store data for different group types
        self.available_groups = []
        
        print("🏢 GROUPS DASHBOARD - SECTORS & INDUSTRIES ANALYSIS")
        print("="*65)
        print("📊 Target columns: Name, Market Cap, Performance metrics")
        print("📊 Group types: Sector, Industry, and others (auto-discovery)")
        print("="*65)
        
        if GROUP_MODULES_AVAILABLE:
            self.group_performance_screener = GroupPerformance()
            self.group_overview_screener = GroupOverview()
            self.group_valuation_screener = GroupValuation()
            
            # Discover available group types
            self.discover_available_groups()
        else:
            print("❌ Group modules not available - using alternative approach")

    def discover_available_groups(self):
        """
        Discover what group types are available in finvizfinance
        """
        print("\n🔍 Discovering available group types...")
        
        # Common group types to test
        potential_groups = [
            'Sector', 'Industry', 'Country', 'Market Cap', 
            'Exchange', 'Index', 'Capitalization'
        ]
        
        self.available_groups = []
        
        for group_type in potential_groups:
            try:
                print(f"   Testing: {group_type}...")
                
                # Try to get a small sample to test if the group type works
                test_data = self.group_performance_screener.screener_view(
                    group=group_type,
                    order='Name'
                )
                
                if test_data is not None and not test_data.empty:
                    self.available_groups.append(group_type)
                    print(f"   ✅ {group_type} - {len(test_data)} groups found")
                else:
                    print(f"   ❌ {group_type} - no data")
                    
                time.sleep(0.5)  # Be gentle with API
                
            except Exception as e:
                print(f"   ❌ {group_type} - error: {str(e)[:50]}")
                continue
        
        if self.available_groups:
            print(f"\n✅ Available group types: {', '.join(self.available_groups)}")
        else:
            print("\n❌ No group types discovered")
        
        return self.available_groups

    def fetch_group_performance_data(self, group='Sector', order='Name', max_retries=3):
        """
        Fetch performance data at the group level (Sector, Industry, etc.)
        """
        if not GROUP_MODULES_AVAILABLE:
            print("❌ Group performance module not available")
            return None
            
        print(f"\n📈 Fetching {group} performance data...")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                group_performance = self.group_performance_screener.screener_view(
                    group=group,
                    order=order
                )
                
                if group_performance is not None and not group_performance.empty:
                    print(f"✅ {group} performance data: {len(group_performance)} groups")
                    print(f"📊 Available columns: {list(group_performance.columns)}")
                    return group_performance
                else:
                    print(f"❌ Attempt {attempt + 1} failed - no data")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        print(f"❌ Failed to fetch {group} performance data after all retries")
        return None

    def fetch_group_overview_data(self, group='Sector', order='Name', max_retries=3):
        """
        Fetch overview data at the group level to get Market Cap
        """
        if not GROUP_MODULES_AVAILABLE:
            print("❌ Group overview module not available")
            return None
            
        print(f"\n📋 Fetching {group} overview data...")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                group_overview = self.group_overview_screener.screener_view(
                    group=group,
                    order=order
                )
                
                if group_overview is not None and not group_overview.empty:
                    print(f"✅ {group} overview data: {len(group_overview)} groups")
                    print(f"📊 Available columns: {list(group_overview.columns)}")
                    return group_overview
                else:
                    print(f"❌ Attempt {attempt + 1} failed - no data")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        print(f"❌ Failed to fetch {group} overview data after all retries")
        return None

    def _standardize_performance_columns(self, data):
        """
        Standardize performance columns to consistent percentage format
        """
        performance_cols = [
            'Perf Week', 'Perf Month', 'Perf Quart', 'Perf Half', 'Perf Year', 'Perf YTD',
            'Performance (Week)', 'Performance (Month)', 'Performance (Quarter)', 
            'Performance (Half Year)', 'Performance (Year)', 'Performance (Year To Date)'
        ]
        
        for col in performance_cols:
            if col in data.columns:
                # Convert to numeric, handling both percentage strings and decimals
                def convert_perf_value(val):
                    if pd.isna(val):
                        return val
                    
                    val_str = str(val).strip()
                    
                    # If it already has %, extract the number
                    if '%' in val_str:
                        try:
                            return float(val_str.replace('%', '').replace(',', ''))
                        except:
                            return val
                    else:
                        # Assume it's a decimal (e.g., 0.05 for 5%)
                        try:
                            numeric_val = float(val_str.replace(',', ''))
                            # If the value is between -1 and 1, it's likely a decimal that should be converted to percentage
                            if -1 <= numeric_val <= 1:
                                return numeric_val * 100
                            else:
                                # If it's already a larger number, assume it's already in percentage format
                                return numeric_val
                        except:
                            return val
                
                data[col] = data[col].apply(convert_perf_value)
        
        return data

    def merge_group_data(self, group_performance, group_overview):
        """
        Merge performance and overview group data and clean up column names
        """
        if group_performance is None and group_overview is None:
            print("❌ No group data to merge")
            return None
        
        # If we have both datasets, merge them
        if group_performance is not None and group_overview is not None:
            print("\n🔄 Merging group performance and overview data...")
            
            merged_data = pd.merge(
                group_performance,
                group_overview,
                on='Name',
                how='outer',
                suffixes=('', '_overview')  # Keep performance columns as primary
            )
            
            # Clean up duplicate columns - prefer performance data over overview
            columns_to_drop = []
            for col in merged_data.columns:
                if col.endswith('_overview'):
                    base_col = col.replace('_overview', '')
                    if base_col in merged_data.columns:
                        # Keep the performance version, drop the overview version
                        columns_to_drop.append(col)
            
            if columns_to_drop:
                print(f"🔄 Removing duplicate columns: {columns_to_drop}")
                merged_data = merged_data.drop(columns=columns_to_drop)
            
        else:
            # Use whichever data we have
            merged_data = group_performance or group_overview
        
        # Standardize performance columns before renaming
        merged_data = self._standardize_performance_columns(merged_data)
        
        # Remove unwanted columns
        unwanted_columns = [
            'Recom', 'Avg Volume', 'Rel Volume', 'Change', 'Volume', 'Stocks', 
            'Dividend', 'P/E', 'Fwd P/E', 'PEG', 'Float Short'
        ]
        
        columns_to_drop = [col for col in unwanted_columns if col in merged_data.columns]
        if columns_to_drop:
            print(f"🔄 Removing unwanted columns: {columns_to_drop}")
            merged_data = merged_data.drop(columns=columns_to_drop)
        
        # Rename columns to match our target format
        column_mapping = {
            'Perf Week': 'Performance (Week)',
            'Perf Month': 'Performance (Month)', 
            'Perf Quart': 'Performance (Quarter)',
            'Perf Half': 'Performance (Half Year)',
            'Perf Year': 'Performance (Year)',
            'Perf YTD': 'Performance (Year To Date)',
            'Market Cap': 'Market Cap.'
        }
        
        merged_data = merged_data.rename(columns=column_mapping)
        
        print(f"✅ Merged group data: {len(merged_data)} groups")
        print(f"📊 Final columns: {list(merged_data.columns)}")
        
        return merged_data

    def create_groups_data_alternative(self, group_type='Sector'):
        """
        Alternative approach: Aggregate individual stock data to create group-level data
        This is a fallback if group modules aren't available
        """
        print(f"\n🔄 Using alternative approach - aggregating individual stock data by {group_type}...")
        
        try:
            # Import the existing dashboard to get stock-level data
            from strength_dashboard import EnhancedStrengthDashboard
            
            # Create an instance and get data
            dashboard = EnhancedStrengthDashboard()
            dashboard.fetch_performance_data(limit=3000)
            dashboard.fetch_overview_data(limit=8000)
            merged_data = dashboard.merge_datasets()
            
            if merged_data is None or merged_data.empty:
                print("❌ No stock data available for aggregation")
                return None
            
            processed_data = dashboard.process_strength_metrics()
            
            if processed_data is None or group_type not in processed_data.columns:
                print(f"❌ No {group_type} data available for aggregation")
                print(f"📊 Available columns: {list(processed_data.columns)}")
                return None
            
            print(f"📊 Aggregating {len(processed_data)} stocks into {group_type} data...")
            
            # Aggregate by the specified group type
            group_metrics = self._aggregate_to_group_level(processed_data, group_type)
            
            return group_metrics
            
        except Exception as e:
            print(f"❌ Alternative approach failed: {e}")
            return None

    def _aggregate_to_group_level(self, stock_data, group_column):
        """
        Aggregate individual stock data to group level (Sector, Industry, etc.)
        """
        print(f"🔄 Aggregating stock data to {group_column} level...")
        
        # Performance columns to aggregate
        perf_columns = ['Perf Week', 'Perf Month', 'Perf Quart', 'Perf Half', 'Perf Year', 'Perf YTD']
        available_perf_cols = [col for col in perf_columns if col in stock_data.columns]
        
        # Prepare aggregation dictionary
        agg_dict = {}
        
        # Performance metrics - use mean
        for col in available_perf_cols:
            agg_dict[col] = 'mean'
        
        # Market cap - sum for total group market cap
        if 'Market Cap' in stock_data.columns:
            # Convert market cap to numeric for proper aggregation
            stock_data['Market Cap Numeric'] = self._convert_market_cap_to_numeric(stock_data['Market Cap'])
            agg_dict['Market Cap Numeric'] = 'sum'
        
        # Count of stocks in each group
        agg_dict['Ticker'] = 'count'
        
        # Perform aggregation
        group_data = stock_data.groupby(group_column).agg(agg_dict).round(2)
        
        # Rename columns for clarity
        column_mapping = {
            'Ticker': 'Stock_Count',
            'Market Cap Numeric': 'Market Cap.'
        }
        
        # Rename performance columns to match target format
        for col in available_perf_cols:
            if col == 'Perf Week':
                column_mapping[col] = 'Performance (Week)'
            elif col == 'Perf Month':
                column_mapping[col] = 'Performance (Month)'
            elif col == 'Perf Quart':
                column_mapping[col] = 'Performance (Quarter)'
            elif col == 'Perf Half':
                column_mapping[col] = 'Performance (Half Year)'
            elif col == 'Perf Year':
                column_mapping[col] = 'Performance (Year)'
            elif col == 'Perf YTD':
                column_mapping[col] = 'Performance (Year To Date)'
        
        group_data = group_data.rename(columns=column_mapping)
        
        # Add Name column (group name from index)
        group_data.reset_index(inplace=True)
        group_data.rename(columns={group_column: 'Name'}, inplace=True)
        
        # Format market cap for display
        if 'Market Cap.' in group_data.columns:
            group_data['Market Cap.'] = group_data['Market Cap.'].apply(self._format_market_cap)
        
        # Sort by overall performance (if available) or by name
        if 'Performance (Month)' in group_data.columns:
            group_data = group_data.sort_values('Performance (Month)', ascending=False)
        else:
            group_data = group_data.sort_values('Name')
        
        print(f"✅ Created {group_column}-level data for {len(group_data)} groups")
        
        return group_data

    def _convert_market_cap_to_numeric(self, market_cap_series):
        """Convert market cap strings to numeric values in millions"""
        def convert_single_market_cap(value):
            if pd.isna(value) or value == '-':
                return 0
            
            value_str = str(value).replace(',', '').replace('$', '')
            
            try:
                if 'T' in value_str.upper():
                    return float(value_str.replace('T', '').replace('t', '')) * 1000000  # Trillions to millions
                elif 'B' in value_str.upper():
                    return float(value_str.replace('B', '').replace('b', '')) * 1000  # Billions to millions
                elif 'M' in value_str.upper():
                    return float(value_str.replace('M', '').replace('m', ''))  # Already in millions
                else:
                    return float(value_str) / 1000000  # Convert raw value to millions
            except:
                return 0
        
        return market_cap_series.apply(convert_single_market_cap)

    def _format_market_cap(self, value):
        """Format market cap for display"""
        if value >= 1000000:
            return f"${value/1000000:.1f}T"
        elif value >= 1000:
            return f"${value/1000:.1f}B"
        else:
            return f"${value:.1f}M"

    def get_leading_groups(self, group_data, sort_by='Performance (Month)', top_n=None):
        """
        Get leading groups sorted by specified metric
        """
        if group_data is None:
            print("❌ No group data available")
            return None
        
        df = group_data.copy()
        
        # Sort by the specified column
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        else:
            print(f"⚠️ Column '{sort_by}' not found. Available columns: {list(df.columns)}")
            return df
        
        if top_n:
            df = df.head(top_n)
        
        return df

    def display_groups_summary(self, group_type, group_data):
        """
        Display a comprehensive summary of the groups
        """
        if group_data is None:
            print(f"❌ No {group_type} data to display")
            return
        
        print(f"\n🏢 LEADING {group_type.upper()}S DASHBOARD")
        print("="*80)
        
        # Display only our target columns (cleaned up)
        target_display_cols = [
            'Name', 'Market Cap.', 'Performance (Week)', 'Performance (Month)',
            'Performance (Quarter)', 'Performance (Half Year)', 'Performance (Year)',
            'Performance (Year To Date)'
        ]
        
        # Get available columns from our target list
        available_cols = [col for col in target_display_cols if col in group_data.columns]
        
        if available_cols:
            print(f"📊 Displaying {len(available_cols)} columns: {', '.join(available_cols)}")
            print("-" * 80)
            
            # Get top groups sorted by monthly performance
            top_groups = self.get_leading_groups(group_data, 'Performance (Month)', top_n=15)
            
            if top_groups is not None:
                # Format the performance columns for better display
                display_data = top_groups[available_cols].copy()
                
                # Format performance columns to show as percentages
                for col in available_cols:
                    if 'Performance' in col and col in display_data.columns:
                        display_data[col] = display_data[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                
                print(display_data.to_string(index=False))
            else:
                print(f"❌ No {group_type} data available")
                
            # Also show some summary stats
            if 'Performance (Month)' in group_data.columns:
                best_performer = group_data.loc[group_data['Performance (Month)'].idxmax()]
                worst_performer = group_data.loc[group_data['Performance (Month)'].idxmin()]
                avg_performance = group_data['Performance (Month)'].mean()
                
                print(f"\n📈 MONTHLY PERFORMANCE SUMMARY:")
                print(f"🥇 Best: {best_performer['Name']} ({best_performer['Performance (Month)']:+.2f}%)")
                print(f"🥉 Worst: {worst_performer['Name']} ({worst_performer['Performance (Month)']:+.2f}%)")
                print(f"📊 Average: {avg_performance:+.2f}%")
        else:
            print("❌ None of the target columns are available")
            print(f"📊 Available columns: {list(group_data.columns)}")

    def export_groups_data(self, group_type, group_data, filename=None):
        """
        Export groups data to CSV (with cleaned columns)
        """
        if group_data is None:
            print(f"❌ No {group_type} data to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"groups_dashboard_{group_type.lower()}s_{timestamp}.csv"
        
        try:
            # Export only the target columns
            target_export_cols = [
                'Name', 'Market Cap.', 'Performance (Week)', 'Performance (Month)',
                'Performance (Quarter)', 'Performance (Half Year)', 'Performance (Year)',
                'Performance (Year To Date)'
            ]
            
            # Get available columns from our target list
            available_cols = [col for col in target_export_cols if col in group_data.columns]
            
            if available_cols:
                export_data = group_data[available_cols].copy()
                export_data.to_csv(filename, index=False)
                print(f"✅ {group_type} data exported to: {filename}")
                print(f"📊 Exported columns: {', '.join(available_cols)}")
            else:
                print(f"❌ No target columns available for export")
            
            return filename
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None

    def analyze_group_type(self, group='Sector', use_alternative=True):
        """
        Analyze a specific group type (Sector, Industry, etc.)
        """
        print(f"\n🚀 Analyzing {group} data...")
        
        group_data = None
        
        if GROUP_MODULES_AVAILABLE and (not hasattr(self, 'available_groups') or group in self.available_groups):
            # Try using the group modules first
            performance_data = self.fetch_group_performance_data(group=group)
            overview_data = self.fetch_group_overview_data(group=group)
            
            if performance_data is not None or overview_data is not None:
                group_data = self.merge_group_data(performance_data, overview_data)
        
        # If group modules don't work or aren't available, use alternative
        if group_data is None and use_alternative:
            print(f"\n🔄 Using alternative approach for {group} (stock aggregation)...")
            group_data = self.create_groups_data_alternative(group_type=group)
        
        if group_data is not None:
            # Store the data
            self.group_data[group] = group_data
            
            # Display and export
            self.display_groups_summary(group, group_data)
            self.export_groups_data(group, group_data)
            
            return group_data
        else:
            print(f"❌ {group} analysis failed")
            return None

    def run_all_groups_analysis(self, groups_to_analyze=None, use_alternative=True):
        """
        Run analysis for multiple group types
        """
        print(f"\n🚀 Starting Multi-Group Dashboard Analysis...")
        
        if groups_to_analyze is None:
            # Default groups to analyze
            groups_to_analyze = ['Sector', 'Industry']
            
            # Add any other discovered groups
            if hasattr(self, 'available_groups') and self.available_groups:
                for group in self.available_groups:
                    if group not in groups_to_analyze:
                        groups_to_analyze.append(group)
        
        print(f"📊 Groups to analyze: {', '.join(groups_to_analyze)}")
        
        results = {}
        
        for group in groups_to_analyze:
            try:
                result = self.analyze_group_type(group, use_alternative)
                if result is not None:
                    results[group] = result
                    print(f"✅ {group} analysis complete")
                else:
                    print(f"❌ {group} analysis failed")
            except Exception as e:
                print(f"❌ {group} analysis error: {e}")
                continue
        
        if results:
            print(f"\n✅ Multi-Group Analysis Complete!")
            print(f"📊 Successfully analyzed: {', '.join(results.keys())}")
            
            # Summary of all groups
            self.display_all_groups_summary()
            
        return results

    def display_all_groups_summary(self):
        """
        Display a summary of all analyzed groups
        """
        if not self.group_data:
            print("❌ No group data available")
            return
        
        print(f"\n📊 MULTI-GROUP SUMMARY")
        print("="*60)
        
        for group_type, data in self.group_data.items():
            if data is not None:
                print(f"\n🏢 {group_type.upper()} ({len(data)} groups)")
                print("-" * 40)
                
                # Show top 5 by monthly performance
                if 'Performance (Month)' in data.columns:
                    top_5 = data.nlargest(5, 'Performance (Month)')
                    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                        name = row['Name']
                        perf = row.get('Performance (Month)', 0)
                        print(f"  {idx}. {name}: {perf:+.2f}%")
                else:
                    # Just show first 5 names
                    for idx, name in enumerate(data['Name'].head(5), 1):
                        print(f"  {idx}. {name}")

    def get_group_comparison(self, metric='Performance (Month)'):
        """
        Compare performance across different group types
        """
        if not self.group_data:
            print("❌ No group data available for comparison")
            return None
        
        comparison_data = []
        
        for group_type, data in self.group_data.items():
            if data is not None and metric in data.columns:
                # Get stats for this group type
                stats = {
                    'Group_Type': group_type,
                    'Count': len(data),
                    'Best_Performer': data.loc[data[metric].idxmax(), 'Name'],
                    'Best_Performance': data[metric].max(),
                    'Worst_Performer': data.loc[data[metric].idxmin(), 'Name'],
                    'Worst_Performance': data[metric].min(),
                    'Average_Performance': data[metric].mean(),
                    'Median_Performance': data[metric].median()
                }
                comparison_data.append(stats)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            print(f"\n📊 GROUP TYPE COMPARISON ({metric})")
            print("="*80)
            print(comparison_df.to_string(index=False))
            
            return comparison_df
        else:
            print(f"❌ No data available for metric: {metric}")
            return None

# Main execution functions
def run_sector_analysis():
    """Run analysis for Sectors only"""
    dashboard = GroupsDashboard()
    return dashboard.analyze_group_type('Sector')

def run_industry_analysis():
    """Run analysis for Industries only"""
    dashboard = GroupsDashboard()
    return dashboard.analyze_group_type('Industry')

def run_groups_dashboard():
    """Run analysis for all available group types"""
    dashboard = GroupsDashboard()
    return dashboard.run_all_groups_analysis()

# Example usage
if __name__ == "__main__":
    print("🏢 Starting Enhanced Groups Dashboard - Sectors & Industries Analysis")
    print("Target: Name, Market Cap, Performance metrics")
    
    # Run the full analysis
    groups_dashboard = GroupsDashboard()
    results = groups_dashboard.run_all_groups_analysis()
    
    if results:
        print("\n✅ Groups Dashboard Analysis Complete!")
        
        print("\n🔧 AVAILABLE FUNCTIONS:")
        print("=" * 50)
        print("# Get specific group data:")
        print("sectors = groups_dashboard.group_data['Sector']")
        print("industries = groups_dashboard.group_data['Industry']")
        print("\n# Compare group types:")
        print("comparison = groups_dashboard.get_group_comparison('Performance (Month)')")
        print("\n# Get leading groups:")
        print("top_sectors = groups_dashboard.get_leading_groups(sectors, 'Performance (Month)', 10)")
        print("\n# Export specific group:")
        print("groups_dashboard.export_groups_data('Sector', sectors, 'my_sectors.csv')")
    else:
        print("❌ Groups Dashboard Analysis failed")