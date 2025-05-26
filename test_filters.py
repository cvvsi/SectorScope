#!/usr/bin/env python3
"""
Test script to check available filters and their values
Save as: test_filters.py
"""

from finvizfinance.screener.performance import Performance

def test_available_filters():
    """Test what filters and values are available"""
    
    print("🔍 Testing available filters...")
    
    try:
        performance_screener = Performance()
        
        # Test different market cap filter values
        market_cap_options = [
            'Small+ (over $300M)',
            'Mid+ (over $2B)', 
            'Large+ (over $10B)',
            'Mega ($200B and more)',
            '+Small (over $300M)',
            'Small',
            'Mid',
            'Large',
            'Mega',
            'Over $300M',
            '$300M+',
            'Small Cap+',
            'Small Cap and above'
        ]
        
        print("\n🧪 Testing Market Cap filter values...")
        
        for option in market_cap_options:
            try:
                print(f"Testing: {option}")
                filters_dict = {'Market Cap.': option}
                performance_screener.set_filter(filters_dict=filters_dict)
                
                # Try to get just 1 result to test if filter works
                result = performance_screener.screener_view(limit=1, verbose=0)
                
                if result is not None and not result.empty:
                    print(f"✅ '{option}' works! Got {len(result)} result(s)")
                    break
                else:
                    print(f"❌ '{option}' returned no results")
                    
            except Exception as e:
                print(f"❌ '{option}' failed: {str(e)[:50]}...")
        
        # Test without market cap filter to see what we get
        print("\n🌐 Testing without Market Cap filter...")
        try:
            performance_screener.set_filter(filters_dict={})
            result = performance_screener.screener_view(limit=10, verbose=0)
            
            if result is not None and not result.empty:
                print(f"✅ No filter works! Got {len(result)} results")
                print("\nSample data columns:")
                print(result.columns.tolist())
                
                if 'Market Cap' in result.columns:
                    print(f"\nSample Market Cap values:")
                    print(result['Market Cap'].head(10).tolist())
            
        except Exception as e:
            print(f"❌ No filter test failed: {e}")
            
    except Exception as e:
        print(f"❌ Error in test: {e}")

def test_simple_fetch():
    """Simple test to fetch data without complex filters"""
    
    print("\n🚀 Testing simple data fetch...")
    
    try:
        performance_screener = Performance()
        
        # Try with minimal or no filters
        print("Fetching data with no filters...")
        result = performance_screener.screener_view(limit=50, verbose=1)
        
        if result is not None and not result.empty:
            print(f"✅ Success! Got {len(result)} stocks")
            print(f"Columns: {list(result.columns)}")
            
            # Check for market cap info
            if 'Market Cap' in result.columns:
                print(f"\nMarket Cap examples:")
                print(result[['Ticker', 'Company', 'Market Cap']].head(10))
            
            # Check for performance columns
            perf_cols = [col for col in result.columns if 'Perf' in col or 'Change' in col]
            print(f"\nPerformance columns found: {perf_cols}")
            
            return result
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Simple fetch failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 FINVIZ FILTER TESTING")
    print("=" * 50)
    
    # Test filters
    test_available_filters()
    
    # Test simple fetch
    sample_data = test_simple_fetch()
    
    if sample_data is not None:
        print(f"\n✅ Basic data fetch successful!")
        print("You can now modify the main script with working filter values.")
    else:
        print(f"\n❌ Unable to fetch data. Check internet connection or FinViz availability.")