"""
UIDAI HACKATHON 2026 - COMPLETE PIPELINE RUNNER
One-command execution of entire UFI analysis
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import AadhaarDataLoader
from feature_engineering import UFIFeatureEngine
from ufi_calculator import UFICalculator
from insight_generator import UFIInsightGenerator
from visualizations import UFIVisualizer

import time

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_complete_pipeline(sample_size='500000', force_recalc=False):
    """
    Execute complete UFI analysis pipeline
    
    Args:
        sample_size: Dataset sample size to use
        force_recalc: Force recalculation even if files exist
    """
    
    start_time = time.time()
    
    print_header("🚀 UIDAI UPDATE FRICTION INDEX (UFI) ANALYSIS PIPELINE")
    
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # ========================
    # STEP 1: DATA LOADING
    # ========================
    print_header("STEP 1: DATA LOADING")
    
    loader = AadhaarDataLoader(data_dir='data/raw')
    enrol_df, bio_df, demo_df = loader.load_all(sample_size=sample_size)
    
    step1_time = time.time()
    print(f"\n⏱️  Step 1 completed in {step1_time - start_time:.2f} seconds")
    
    # ========================
    # STEP 2: FEATURE ENGINEERING
    # ========================
    print_header("STEP 2: UFI COMPONENT CALCULATION")
    
    components_path = 'data/processed/ufi_components.csv'
    
    if not force_recalc and Path(components_path).exists():
        print(f"📁 Loading existing components from {components_path}")
        import pandas as pd
        ufi_components = pd.read_csv(components_path)
    else:
        print("🔧 Calculating UFI components...")
        engine = UFIFeatureEngine(enrol_df, bio_df, demo_df)
        ufi_components = engine.calculate_all_components()
        ufi_components.to_csv(components_path, index=False)
        print(f"💾 Saved components to {components_path}")
    
    step2_time = time.time()
    print(f"\n⏱️  Step 2 completed in {step2_time - step1_time:.2f} seconds")
    
    # ========================
    # STEP 3: UFI CALCULATION
    # ========================
    print_header("STEP 3: COMPOSITE UFI CALCULATION")
    
    calculator = UFICalculator(ufi_components)
    ufi_results = calculator.compute_ufi(weighting_method='pca')
    
    # Export results
    calculator.export_results(output_dir='data/processed')
    
    step3_time = time.time()
    print(f"\n⏱️  Step 3 completed in {step3_time - step2_time:.2f} seconds")
    
    # ========================
    # STEP 4: INSIGHT GENERATION
    # ========================
    print_header("STEP 4: INSIGHT GENERATION")
    
    generator = UFIInsightGenerator(ufi_scores_path='data/processed/ufi_scores.csv')
    insights = generator.generate_all_insights()
    generator.export_insights(output_dir='outputs/reports')
    
    step4_time = time.time()
    print(f"\n⏱️  Step 4 completed in {step4_time - step3_time:.2f} seconds")
    
    # ========================
    # STEP 5: VISUALIZATION
    # ========================
    print_header("STEP 5: VISUALIZATION GENERATION")
    
    visualizer = UFIVisualizer(ufi_scores_path='data/processed/ufi_scores.csv')
    viz_outputs = visualizer.generate_all_visualizations()
    
    step5_time = time.time()
    print(f"\n⏱️  Step 5 completed in {step5_time - step4_time:.2f} seconds")
    
    # ========================
    # PIPELINE SUMMARY
    # ========================
    print_header("✅ PIPELINE EXECUTION COMPLETE")
    
    total_time = time.time() - start_time
    
    print("📊 EXECUTION SUMMARY:")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Districts analyzed: {len(ufi_results)}")
    print(f"   Mean UFI: {ufi_results['UFI'].mean():.2f}")
    print(f"   High friction districts: {(ufi_results['UFI'] > 75).sum()}")
    
    print("\n📁 OUTPUT FILES GENERATED:")
    print(f"   ✅ data/processed/ufi_components.csv")
    print(f"   ✅ data/processed/ufi_scores.csv")
    print(f"   ✅ data/processed/top_friction_districts.csv")
    print(f"   ✅ data/processed/low_friction_districts.csv")
    print(f"   ✅ data/processed/state_ufi_summary.csv")
    print(f"   ✅ outputs/reports/ufi_insights.json")
    print(f"   ✅ outputs/reports/ufi_insights_report.txt")
    
    print("\n🎨 VISUALIZATIONS GENERATED:")
    for name, path in viz_outputs.items():
        if path:
            print(f"   ✅ {path}")
    
    print("\n" + "="*70)
    print("🎉 READY FOR SUBMISSION!")
    print("="*70)
    
    return {
        'ufi_results': ufi_results,
        'insights': insights,
        'visualizations': viz_outputs,
        'execution_time': total_time
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete UFI analysis pipeline')
    parser.add_argument('--sample-size', default='500000', help='Dataset sample size')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    
    args = parser.parse_args()
    
    results = run_complete_pipeline(
        sample_size=args.sample_size,
        force_recalc=args.force
    )