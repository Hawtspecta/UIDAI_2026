"""
UIDAI HACKATHON 2026 - FEATURE ENGINEERING MODULE
UFI Component Calculation Engine - FINAL CORRECTED VERSION

VERIFIED COLUMNS:
- Enrollment: age_0_5, age_5_17, age_18_greater
- Biometric: bio_age_5_17, bio_age_17_
- Demographic: demo_age_5_17, demo_age_17_

The 5 Components of Update Friction Index (UFI):
1. Demographic Update Intensity (Fluidity Score)
2. Biometric Refresh Rate (Momentum)
3. Age Group Update Disparity (Digital Age Gap)
4. Update-to-Enrollment Ratio (System Load)
5. Temporal Volatility (Change Detection)
"""

import pandas as pd
import numpy as np
from scipy import stats

class UFIFeatureEngine:
    """Calculate all 5 UFI component metrics"""
    
    def __init__(self, enrollment_df, biometric_df, demographic_df):
        self.enrol = enrollment_df.copy()
        self.bio = biometric_df.copy()
        self.demo = demographic_df.copy()
        
    def calculate_demographic_update_intensity(self):
        """
        Component 1: Demographic Update Intensity
        Measures socioeconomic mobility through address/detail changes
        
        Formula: Total demographic updates / Total enrollments by district
        High value = High mobility/fluidity
        """
        print("🔧 Calculating Component 1: Demographic Update Intensity...")
        
        # Aggregate demographic updates by district
        demo_agg = self.demo.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        
        demo_agg['total_demo_updates'] = (
            demo_agg['demo_age_5_17'] + demo_agg['demo_age_17_']
        )
        
        # Aggregate enrollments by district
        enrol_agg = self.enrol.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        enrol_agg['total_enrollments'] = (
            enrol_agg['age_0_5'] + 
            enrol_agg['age_5_17'] + 
            enrol_agg['age_18_greater']
        )
        
        # Merge and calculate intensity
        merged = pd.merge(
            demo_agg[['state', 'district', 'total_demo_updates']],
            enrol_agg[['state', 'district', 'total_enrollments']],
            on=['state', 'district'],
            how='outer'
        ).fillna(0)

        MIN_ENROLLMENTS = 100
        before = len(merged)
        merged = merged[merged['total_enrollments'] >= MIN_ENROLLMENTS]
        after = len(merged)

        if before - after > 0:
            print(f"   🧹 Dropped {before - after} low-population districts (<{MIN_ENROLLMENTS} enrollments)")
        
        merged['demo_update_intensity'] = np.where(
            merged['total_enrollments'] > 0,
            (merged['total_demo_updates'] / merged['total_enrollments']) * 100,
            0
        )
        
        print(f"   ✅ Range: {merged['demo_update_intensity'].min():.2f} - {merged['demo_update_intensity'].max():.2f}")
        print(f"   ✅ Mean: {merged['demo_update_intensity'].mean():.2f}")
        
        return merged[['state', 'district', 'demo_update_intensity', 'total_enrollments']]
    
    def calculate_biometric_refresh_rate(self):
        """
        Component 2: Biometric Refresh Rate
        Measures security awareness and aging population dynamics
        
        Formula: Monthly growth rate of biometric updates
        High value = Accelerating biometric adoption
        """
        print("🔧 Calculating Component 2: Biometric Refresh Rate...")
        
        # Ensure date is datetime
        self.bio['date'] = pd.to_datetime(self.bio['date'], format='%d-%m-%Y', errors='coerce')
        self.bio['year_month'] = self.bio['date'].dt.to_period('M')
        
        # Calculate monthly updates by district
        bio_monthly = self.bio.groupby(['state', 'district', 'year_month']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        
        bio_monthly['total_bio_updates'] = (
            bio_monthly['bio_age_5_17'] + bio_monthly['bio_age_17_']
        )
        
        # Calculate month-over-month growth rate
        bio_monthly = bio_monthly.sort_values(['state', 'district', 'year_month'])
        bio_monthly['bio_refresh_rate'] = bio_monthly.groupby(['state', 'district'])['total_bio_updates'].pct_change() * 100
        
        # Aggregate to district level (mean growth rate)
        bio_rate = bio_monthly.groupby(['state', 'district'])['bio_refresh_rate'].mean().reset_index()
        bio_rate['bio_refresh_rate'] = bio_rate['bio_refresh_rate'].fillna(0)
        
        # Replace inf values with 0
        bio_rate['bio_refresh_rate'] = bio_rate['bio_refresh_rate'].replace([np.inf, -np.inf], 0)
        
        print(f"   ✅ Range: {bio_rate['bio_refresh_rate'].min():.2f} - {bio_rate['bio_refresh_rate'].max():.2f}")
        print(f"   ✅ Mean: {bio_rate['bio_refresh_rate'].mean():.2f}")
        
        return bio_rate
    
    def calculate_age_group_disparity(self):
        """
        Component 3: Age Group Update Disparity (Digital Age Gap)
        Measures intergenerational digital divide
        
        Formula: |Young update rate - Elder update rate|
        High value = High generational inequality
        """
        print("🔧 Calculating Component 3: Age Group Update Disparity...")

        # Aggregate by district
        demo_age = self.demo.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        
        enrol_age = self.enrol.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        merged = pd.merge(demo_age, enrol_age, on=['state', 'district'], how='outer').fillna(0)
        
        # Calculate rates - align age groups properly
        # demo_age_5_17 should be compared to age_5_17 enrollments
        merged['young_update_rate'] = np.where(
            merged['age_5_17'] > 0,
            (merged['demo_age_5_17'] / merged['age_5_17']) * 100,
            0
        )
        
        merged['elder_update_rate'] = np.where(
            merged['age_18_greater'] > 0,
            (merged['demo_age_17_'] / merged['age_18_greater']) * 100,
            0
        )

        # Aggregate total enrollments per district
        merged['total_enrollments'] = merged[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)

        MIN_ENROLLMENTS = 100
        before = len(merged)
        merged = merged[merged['total_enrollments'] >= MIN_ENROLLMENTS]
        after = len(merged)

        if before - after > 0:
            print(f"   🧹 Dropped {before - after} low-population districts (<{MIN_ENROLLMENTS} enrollments)")
        
        merged['age_disparity'] = abs(merged['young_update_rate'] - merged['elder_update_rate'])
        
        print(f"   ✅ Range: {merged['age_disparity'].min():.2f} - {merged['age_disparity'].max():.2f}")
        print(f"   ✅ Mean: {merged['age_disparity'].mean():.2f}")
        
        return merged[['state', 'district', 'age_disparity']]
    
    def calculate_update_enrollment_ratio(self):
        """
        Component 4: Update-to-Enrollment Ratio
        Measures system load and access patterns
        
        Formula: (Total updates / Total enrollments) by district
        High value = Heavy system usage OR infrastructure gaps
        """
        print("🔧 Calculating Component 4: Update-to-Enrollment Ratio...")
        
        # Total updates (demo + bio)
        demo_total = self.demo.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_total['total_demo'] = demo_total['demo_age_5_17'] + demo_total['demo_age_17_']
        
        bio_total = self.bio.groupby(['state', 'district']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        bio_total['total_bio'] = bio_total['bio_age_5_17'] + bio_total['bio_age_17_']
        
        # Merge updates
        updates = pd.merge(
            demo_total[['state', 'district', 'total_demo']],
            bio_total[['state', 'district', 'total_bio']],
            on=['state', 'district'],
            how='outer'
        ).fillna(0)
        
        updates['total_updates'] = updates['total_demo'] + updates['total_bio']
        
        # Get enrollments
        enrol_total = self.enrol.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        enrol_total['total_enrollments'] = (
            enrol_total['age_0_5'] + 
            enrol_total['age_5_17'] + 
            enrol_total['age_18_greater']
        )
        
        # Calculate ratio
        merged = pd.merge(
            updates[['state', 'district', 'total_updates']],
            enrol_total[['state', 'district', 'total_enrollments']],
            on=['state', 'district'],
            how='outer'
        ).fillna(0)

        MIN_ENROLLMENTS = 100
        before = len(merged)
        merged = merged[merged['total_enrollments'] >= MIN_ENROLLMENTS]
        after = len(merged)

        if before - after > 0:
            print(f"   🧹 Dropped {before - after} low-population districts (<{MIN_ENROLLMENTS} enrollments)")
        
        merged['update_enrol_ratio'] = np.where(
            merged['total_enrollments'] > 0,
            (merged['total_updates'] / merged['total_enrollments']) * 100,
            0
        )
        
        print(f"   ✅ Range: {merged['update_enrol_ratio'].min():.2f} - {merged['update_enrol_ratio'].max():.2f}")
        print(f"   ✅ Mean: {merged['update_enrol_ratio'].mean():.2f}")
        
        return merged[['state', 'district', 'update_enrol_ratio']]
    
    def calculate_temporal_volatility(self):
        """
        Component 5: Temporal Volatility
        Measures stability vs sudden changes in update patterns
        
        Formula: Coefficient of variation of monthly updates
        High value = High instability/sudden shocks
        """
        print("🔧 Calculating Component 5: Temporal Volatility...")
        
        # Combine all update data with dates
        self.demo['date'] = pd.to_datetime(self.demo['date'], format='%d-%m-%Y', errors='coerce')
        self.demo['year_month'] = self.demo['date'].dt.to_period('M')
        
        demo_monthly = self.demo.groupby(['state', 'district', 'year_month']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_monthly['monthly_updates'] = demo_monthly['demo_age_5_17'] + demo_monthly['demo_age_17_']
        
        # Calculate coefficient of variation by district
        volatility = demo_monthly.groupby(['state', 'district'])['monthly_updates'].agg([
            ('mean', 'mean'),
            ('std', 'std')
        ]).reset_index()
        
        volatility['temporal_volatility'] = np.where(
            volatility['mean'] > 0,
            (volatility['std'] / volatility['mean']) * 100,
            0
        )
        
        # Replace inf values
        volatility['temporal_volatility'] = volatility['temporal_volatility'].replace([np.inf, -np.inf], 0)
        
        print(f"   ✅ Range: {volatility['temporal_volatility'].min():.2f} - {volatility['temporal_volatility'].max():.2f}")
        print(f"   ✅ Mean: {volatility['temporal_volatility'].mean():.2f}")
        
        return volatility[['state', 'district', 'temporal_volatility']]
    
    def calculate_all_components(self):
        """Calculate all 5 UFI components and merge"""
        print("\n" + "="*60)
        print("🚀 CALCULATING ALL UFI COMPONENTS")
        print("="*60 + "\n")
        
        c1 = self.calculate_demographic_update_intensity()
        c2 = self.calculate_biometric_refresh_rate()
        c3 = self.calculate_age_group_disparity()
        c4 = self.calculate_update_enrollment_ratio()
        c5 = self.calculate_temporal_volatility()
        
        # Merge all components
        print("\n🔗 Merging all components...")
        ufi_components = c1.copy()
        
        for df in [c2, c3, c4, c5]:
            ufi_components = pd.merge(
                ufi_components,
                df,
                on=['state', 'district'],
                how='inner'  # Inner join to keep only complete records
            )
        
        ufi_components = ufi_components.fillna(0)
        
        print(f"✅ UFI Components calculated for {len(ufi_components)} districts")
        print(f"\nColumns: {list(ufi_components.columns)}")
        
        return ufi_components


# USAGE EXAMPLE
if __name__ == "__main__":
    from data_loader import AadhaarDataLoader
    import os
    
    # Load data
    print("="*60)
    print("LOADING UIDAI DATASETS")
    print("="*60)
    loader = AadhaarDataLoader(data_dir='data/raw')
    enrol_df, bio_df, demo_df = loader.load_all(sample_size='500000')
    
    # Calculate UFI components
    engine = UFIFeatureEngine(enrol_df, bio_df, demo_df)
    ufi_components = engine.calculate_all_components()
    
    # Save to processed data
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/ufi_components.csv'
    ufi_components.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("COMPONENT STATISTICS")
    print("="*60)
    print(ufi_components.describe())
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*60)