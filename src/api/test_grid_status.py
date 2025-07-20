"""
Test if gridstatus works WITHOUT an API key
Run this first to see what we actually need
"""

def test_gridstatus_no_api_key():
    """Test if we can get ERCOT data without API key"""
    print("🔍 Testing gridstatus without API key...")
    
    try:
        import gridstatus
        print("✅ gridstatus imported successfully")
        
        # Try to initialize ERCOT without API key
        ercot = gridstatus.ERCOT()
        print("✅ ERCOT initialized without API key")
        
        # Try to get recent load data (small sample)
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)  # Just 1 day to test
        
        print(f"🔄 Testing data fetch for {start_date.date()}...")
        
        load_data = ercot.get_load(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if not load_data.empty:
            print("✅ SUCCESS! Got data without API key")
            print(f"   Records: {len(load_data)}")
            print(f"   Columns: {list(load_data.columns)}")
            print("\n📊 Sample data:")
            print(load_data.head(3))
            return True
        else:
            print("⚠️  Got empty data (might be normal for recent dates)")
            return False
            
    except ImportError:
        print("❌ gridstatus not installed")
        print("   Run: pip install gridstatus")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        if "api" in str(e).lower() or "key" in str(e).lower() or "auth" in str(e).lower():
            print("🔑 Looks like we might need an API key")
            return False
        else:
            print("🤔 Different error - might be network or data availability")
            return False

if __name__ == "__main__":
    print("🧪 Testing gridstatus requirements...")
    print("=" * 50)
    
    success = test_gridstatus_no_api_key()
    
    if success:
        print("\n🎉 GREAT! We don't need an API key for basic ERCOT data")
        print("   Your config can be simplified")
    else:
        print("\n🔑 We might need to get an API key")
        print("   Let's check gridstatus documentation")