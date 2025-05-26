# Create this as test_quote.py
from finvizfinance.quote import finvizfinance

# Test with a popular stock
print("🧪 Testing ticker_full_info() data structure...")

try:
    stock = finvizfinance('AAPL')
    data = stock.ticker_full_info()
    
    if data is not None:
        print(f"✅ Success! Data type: {type(data)}")
        
        if hasattr(data, 'columns'):
            print(f"📊 Columns: {list(data.columns)}")
            print(f"📏 Shape: {data.shape}")
            print("\n📋 Sample data:")
            print(data.head())
        else:
            print(f"📊 Data structure: {data}")
            
    else:
        print("❌ No data returned")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Also test ticker_fundament to see what's available
try:
    print("\n🧪 Testing ticker_fundament()...")
    fundament = stock.ticker_fundament()
    print(f"✅ Fundament data: {type(fundament)}")
    
    if isinstance(fundament, dict):
        print(f"📊 Keys: {list(fundament.keys())}")
        # Show a few sample values
        for i, (key, value) in enumerate(fundament.items()):
            if i < 10:  # Show first 10 items
                print(f"  {key}: {value}")
        
except Exception as e:
    print(f"❌ Fundament error: {e}")