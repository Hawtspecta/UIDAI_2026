import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AadhaarDataLoader:
    """
    Unified data loader for all three UIDAI datasets:
    1. Enrollment Data
    2. Biometric Update Data  
    3. Demographic Update Data
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.enrollment_data = None
        self.biometric_data = None
        self.demographic_data = None
        
    def load_enrollment_data(self):
        """Load enrollment dataset"""
        data_path = self.data_dir / 'api_data_aadhar_enrolment'
        files = list(data_path.glob('*.csv'))
        
        if not files:
            raise FileNotFoundError(f"No enrollment files found in: {data_path}")
        
        print(f"📊 Loading enrollment data from {len(files)} files...")
        dfs = []
        for file in files:
            print(f"   Loading: {file.name}")
            df = pd.read_csv(file)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean state/district names
        df['state'] = df['state'].str.strip().str.title()
        df['district'] = df['district'].str.strip().str.title()
        
        self.enrollment_data = df
        print(f"✅ Loaded {len(df):,} enrollment records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   States: {df['state'].nunique()}, Districts: {df['district'].nunique()}")
        
        return df
    
    def load_biometric_data(self):
        """Load biometric update dataset"""
        data_path = self.data_dir / 'api_data_aadhar_biometric'
        files = list(data_path.glob('*.csv'))
        
        if not files:
            raise FileNotFoundError(f"No biometric files found in: {data_path}")
        
        print(f"📊 Loading biometric data from {len(files)} files...")
        dfs = []
        for file in files:
            print(f"   Loading: {file.name}")
            df = pd.read_csv(file)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean state/district names
        df['state'] = df['state'].str.strip().str.title()
        df['district'] = df['district'].str.strip().str.title()
        
        self.biometric_data = df
        print(f"✅ Loaded {len(df):,} biometric update records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_demographic_data(self):
        """Load demographic update dataset"""
        data_path = self.data_dir / 'api_data_aadhar_demographic'
        files = list(data_path.glob('*.csv'))
        
        if not files:
            raise FileNotFoundError(f"No demographic files found in: {data_path}")
        
        print(f"📊 Loading demographic data from {len(files)} files...")
        dfs = []
        for file in files:
            print(f"   Loading: {file.name}")
            df = pd.read_csv(file)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean state/district names
        df['state'] = df['state'].str.strip().str.title()
        df['district'] = df['district'].str.strip().str.title()
        
        self.demographic_data = df
        print(f"✅ Loaded {len(df):,} demographic update records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_all(self):
        """Load all three datasets"""
        print("🚀 Loading all UIDAI datasets...\n")
        
        self.load_enrollment_data()
        self.load_biometric_data()
        self.load_demographic_data()
        
        print("\n✅ ALL DATASETS LOADED SUCCESSFULLY!")
        return self.enrollment_data, self.biometric_data, self.demographic_data
    
    def get_summary_stats(self):
        """Generate summary statistics"""
        stats = {}
        
        if self.enrollment_data is not None:
            stats['enrollment'] = {
                'records': len(self.enrollment_data),
                'date_range': f"{self.enrollment_data['date'].min()} to {self.enrollment_data['date'].max()}",
                'states': self.enrollment_data['state'].nunique(),
                'districts': self.enrollment_data['district'].nunique(),
                'columns': list(self.enrollment_data.columns)
            }
        
        if self.biometric_data is not None:
            stats['biometric'] = {
                'records': len(self.biometric_data),
                'date_range': f"{self.biometric_data['date'].min()} to {self.biometric_data['date'].max()}",
                'states': self.biometric_data['state'].nunique(),
                'districts': self.biometric_data['district'].nunique(),
                'columns': list(self.biometric_data.columns)
            }
        
        if self.demographic_data is not None:
            stats['demographic'] = {
                'records': len(self.demographic_data),
                'date_range': f"{self.demographic_data['date'].min()} to {self.demographic_data['date'].max()}",
                'states': self.demographic_data['state'].nunique(),
                'districts': self.demographic_data['district'].nunique(),
                'columns': list(self.demographic_data.columns)
            }
        
        return stats


# USAGE EXAMPLE
if __name__ == "__main__":
    loader = AadhaarDataLoader(data_dir='data/raw')
    enrol_df, bio_df, demo_df = loader.load_all()
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    stats = loader.get_summary_stats()
    for dataset_name, dataset_stats in stats.items():
        print(f"\n📋 {dataset_name.upper()}")
        for key, value in dataset_stats.items():
            if key != 'columns':
                print(f"   {key}: {value}")