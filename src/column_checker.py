"""
COLUMN VERIFICATION SCRIPT - FIXED FOR NESTED FOLDERS
Checks actual column names in datasets to prevent runtime errors
"""

import pandas as pd
from pathlib import Path

def check_all_columns():
    """Verify column names in all datasets"""
    
    data_dir = Path('data/raw')
    
    print("="*70)
    print("🔍 DATASET COLUMN VERIFICATION")
    print("="*70 + "\n")
    
    # Find all Excel/CSV files recursively
    all_files = list(data_dir.rglob('*.xlsx')) + list(data_dir.rglob('*.xls')) + list(data_dir.rglob('*.csv'))
    
    print(f"Found {len(all_files)} files total\n")
    
    # Group files by type
    enrollment_files = [f for f in all_files if 'enrol' in f.name.lower()]
    biometric_files = [f for f in all_files if 'biometric' in f.name.lower()]
    demographic_files = [f for f in all_files if 'demographic' in f.name.lower()]
    
    files = {
        'ENROLLMENT': enrollment_files,
        'BIOMETRIC': biometric_files,
        'DEMOGRAPHIC': demographic_files
    }
    
    for dataset_type, file_list in files.items():
        if not file_list:
            print(f"❌ No {dataset_type} files found!")
            continue
        
        print(f"\n{'='*70}")
        print(f"📊 {dataset_type} - Found {len(file_list)} file(s)")
        print('='*70)
        
        # Show all files found
        for f in file_list:
            print(f"   📄 {f.relative_to(data_dir)}")
        
        # Use first file for column check
        file_path = file_list[0]
        print(f"\n🔍 Checking columns in: {file_path.name}")
        print("-"*70)
        
        try:
            # Read file (auto-detect CSV vs Excel)
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=5)
            else:
                df = pd.read_excel(file_path, nrows=5)
            
            # Standardize column names (same as data_loader)
            original_cols = list(df.columns)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            standardized_cols = list(df.columns)
            
            print(f"\n✅ ORIGINAL COLUMNS ({len(original_cols)}):")
            for i, col in enumerate(original_cols, 1):
                print(f"   {i:2d}. {col}")
            
            print(f"\n✅ STANDARDIZED COLUMNS ({len(standardized_cols)}):")
            for i, col in enumerate(standardized_cols, 1):
                print(f"   {i:2d}. {col}")
            
            # Show sample data
            print(f"\n📊 SAMPLE DATA (first 3 rows):")
            print(df.head(3).to_string(index=False))
            print("\n")
            
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}\n")
    
    print("="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    check_all_columns()