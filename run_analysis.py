#!/usr/bin/env python3
"""
Simple Runner Script for FinViz Performance Tracker
Save this as: run_analysis.py
"""

def main():
    try:
        print("🚀 Starting FinViz Performance Analysis...")
        
        # Import the main tracker
        from performance_tracker import run_performance_analysis
        
        # Run main analysis
        tracker = run_performance_analysis()
        
        if tracker and tracker.processed_data is not None:
            print("\n✅ Main analysis complete!")
            
            # Try to run advanced analysis
            try:
                from advanced_utils import run_advanced_analysis
                analyzer = run_advanced_analysis(tracker)
                
                if analyzer:
                    print("\n📊 Advanced analysis complete!")
                    
                    # Export results
                    print("\n💾 Exporting results to Excel...")
                    analyzer.export_analysis('finviz_hot_stocks')
                    
                    print("\n🎯 QUICK ACCESS EXAMPLES:")
                    print("=" * 50)
                    print("# Get top 20 overall performers:")
                    print("top_20 = tracker.get_top_performers(20)")
                    print("\n# Get top technology stocks:")
                    print("tech_tops = tracker.get_sector_top_performers('Technology', 15)")
                    print("\n# Get momentum stocks:")
                    print("momentum = analyzer.momentum_analysis(min_score=5.0)")
                    print("\n# Generate watchlist:")
                    print("watchlist = analyzer.generate_watchlist('momentum', 25)")
                    
                    # Interactive mode
                    print("\n🔧 ENTERING INTERACTIVE MODE...")
                    print("You can now use 'tracker' and 'analyzer' objects")
                    print("Type 'exit()' to quit")
                    
                    # Make objects available in global scope for interactive use
                    globals()['tracker'] = tracker
                    globals()['analyzer'] = analyzer
                    
                    # Start interactive Python shell
                    import code
                    code.interact(local=globals())
                    
            except ImportError:
                print("⚠️ Advanced analysis not available (advanced_utils.py not found)")
                print("You can still use the tracker object for basic analysis")
                
                # Interactive mode with just tracker
                globals()['tracker'] = tracker
                import code
                code.interact(local=globals())
        
        else:
            print("❌ Failed to get data. Check your internet connection and try again.")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed: pip install finvizfinance pandas")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that performance_tracker.py is in the same directory")

if __name__ == "__main__":
    main()