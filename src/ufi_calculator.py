"""
UIDAI HACKATHON 2026 - UFI CALCULATOR MODULE
Composite Update Friction Index Construction

Methodology: PCA-based weighting for data-driven component importance
"""

from tokenize import group
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


class UFICalculator:
    """
    Calculate the composite Update Friction Index (UFI)

    UFI = Weighted combination of 5 components:
    - w1 * Demographic Update Intensity
    - w2 * Biometric Refresh Rate
    - w3 * Age Group Disparity
    - w4 * Update-Enrollment Ratio
    - w5 * Temporal Volatility
    """

    def __init__(self, ufi_components_df):
        self.components = ufi_components_df.copy()
        self.feature_cols = [
            "demo_update_intensity",
            "bio_refresh_rate",
            "age_disparity",
            "update_enrol_ratio",
            "temporal_volatility",
        ]
        self.weights = None
        self.scaler = StandardScaler()

    def normalize_components(self):
        """Normalize all components to 0-100 scale"""
        print("🔧 Normalizing UFI components...")

        for col in self.feature_cols:
            if col in self.components.columns:
                # Min-Max normalization to 0-100
                min_val = self.components[col].min()
                max_val = self.components[col].max()

                if max_val - min_val > 0:
                    self.components[f"{col}_normalized"] = (
                        (self.components[col] - min_val) / (max_val - min_val)
                    ) * 100
                else:
                    self.components[f"{col}_normalized"] = 0

        print("   ✅ Normalization complete")

    def calculate_pca_weights(self):
        """
        Use PCA to determine data-driven weights
        First principal component loadings = component importance
        """
        print("🔧 Calculating PCA-based weights...")

        # Prepare data
        normalized_cols = [
            f"{col}_normalized"
            for col in self.feature_cols
            if f"{col}_normalized" in self.components.columns
        ]

        X = self.components[normalized_cols].fillna(0)

        # Standardize for PCA
        X_scaled = self.scaler.fit_transform(X)

        # PCA with 1 component
        pca = PCA(n_components=1)
        pca.fit(X_scaled)

        # Get loadings (weights)
        loadings = np.abs(pca.components_[0])

        # Normalize weights to sum to 1
        self.weights = loadings / loadings.sum()

        print("   ✅ PCA-derived weights:")
        for i, col in enumerate(self.feature_cols):
            print(f"      {col}: {self.weights[i]:.3f}")

        print(f"\n   📊 Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")

        return self.weights

    def calculate_equal_weights(self):
        """Alternative: Equal weighting (simpler, more interpretable)"""
        print("🔧 Using equal weights...")
        self.weights = np.ones(len(self.feature_cols)) / len(self.feature_cols)

        print("   ✅ Equal weights:")
        for i, col in enumerate(self.feature_cols):
            print(f"      {col}: {self.weights[i]:.3f}")

        return self.weights

    def calculate_custom_weights(self, weights_dict):
        """
        Use custom weights specified by user

        Args:
            weights_dict: Dictionary mapping feature names to weights
        """
        print("🔧 Using custom weights...")

        weight_list = []
        for col in self.feature_cols:
            weight_list.append(weights_dict.get(col, 0.2))

        # Normalize to sum to 1
        total = sum(weight_list)
        self.weights = np.array(weight_list) / total

        print("   ✅ Custom weights:")
        for i, col in enumerate(self.feature_cols):
            print(f"      {col}: {self.weights[i]:.3f}")

        return self.weights

    def compute_ufi(self, weighting_method="pca"):
        """
        Compute the final UFI score

        Args:
            weighting_method: 'pca', 'equal', or dict for custom
        """
        print("\n" + "=" * 60)
        print("🚀 COMPUTING UPDATE FRICTION INDEX (UFI)")
        print("=" * 60 + "\n")

        # Normalize components
        self.normalize_components()

        # Apply state-specific normalization first
        self.state_specific_normalization()

        # 🔒 Outlier clipping (critical for PCA stability)
        print("🔧 Clipping extreme values (1%–99%)...")
        for col in self.feature_cols:
            if col in self.components.columns:
                lower = self.components[col].quantile(0.01)
                upper = self.components[col].quantile(0.99)
                self.components[col] = self.components[col].clip(lower, upper)

        print("   ✅ Outlier clipping complete")

        # Calculate weights
        if weighting_method == "pca":
            self.calculate_pca_weights()
        elif weighting_method == "equal":
            self.calculate_equal_weights()
        elif isinstance(weighting_method, dict):
            self.calculate_custom_weights(weighting_method)
        else:
            raise ValueError("Invalid weighting_method. Use 'pca', 'equal', or dict")

        # Compute UFI as weighted sum
        print("\n🔧 Computing weighted UFI scores...")

        normalized_cols = [f"{col}_state_norm" for col in self.feature_cols]

        ufi_scores = np.zeros(len(self.components))

        for i, col in enumerate(normalized_cols):
            if col in self.components.columns:
                ufi_scores += self.weights[i] * self.components[col].fillna(0)

        self.components["UFI"] = ufi_scores

        # Categorize UFI levels
        self.components["UFI_Category"] = pd.cut(
            self.components["UFI"],
            bins=[0, 25, 50, 75, 100],
            labels=[
                "Low Friction",
                "Moderate Friction",
                "High Friction",
                "Very High Friction",
            ],
        )

        # Apply per-capita weighting
        self.apply_per_capita_weighting()

        print("   ✅ UFI computed successfully")
        print(f"\n   📊 UFI Statistics:")
        print(f"      Mean: {self.components['UFI'].mean():.2f}")
        print(f"      Median: {self.components['UFI'].median():.2f}")
        print(f"      Std Dev: {self.components['UFI'].std():.2f}")
        print(f"      Min: {self.components['UFI'].min():.2f}")
        print(f"      Max: {self.components['UFI'].max():.2f}")

        print(f"\n   📊 UFI Category Distribution:")
        print(self.components["UFI_Category"].value_counts().sort_index())

        return self.components

    def get_top_friction_districts(self, n=10):
        """Get top N districts with highest UFI"""
        top = self.components.nlargest(n, "UFI")[
            ["state", "district", "UFI", "UFI_Category"]
            + [f"{col}_normalized" for col in self.feature_cols]
        ]
        return top

    def get_low_friction_districts(self, n=10):
        """Get top N districts with lowest UFI"""
        low = self.components.nsmallest(n, "UFI")[
            ["state", "district", "UFI", "UFI_Category"]
            + [f"{col}_normalized" for col in self.feature_cols]
        ]
        return low

    def get_state_summary(self):
        """Aggregate UFI by state"""
        state_summary = (
            self.components.groupby("state")
            .agg({"UFI": ["mean", "median", "std", "min", "max"], "district": "count"})
            .round(2)
        )

        state_summary.columns = [
            "UFI_Mean",
            "UFI_Median",
            "UFI_StdDev",
            "UFI_Min",
            "UFI_Max",
            "Num_Districts",
        ]
        state_summary = state_summary.sort_values("UFI_Mean", ascending=False)

        return state_summary.reset_index()

    def export_results(self, output_dir="data/processed"):
        """Export UFI results to CSV"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Full dataset
        full_path = f"{output_dir}/ufi_scores.csv"
        self.components.to_csv(full_path, index=False)
        print(f"💾 Saved full UFI dataset to: {full_path}")

        # Top friction
        top_path = f"{output_dir}/top_friction_districts.csv"
        self.get_top_friction_districts(20).to_csv(top_path, index=False)
        print(f"💾 Saved top friction districts to: {top_path}")

        # Low friction
        low_path = f"{output_dir}/low_friction_districts.csv"
        self.get_low_friction_districts(20).to_csv(low_path, index=False)
        print(f"💾 Saved low friction districts to: {low_path}")

        # State summary
        state_path = f"{output_dir}/state_ufi_summary.csv"
        self.get_state_summary().to_csv(state_path, index=False)
        print(f"💾 Saved state summary to: {state_path}")

    def state_specific_normalization(self):
        """
        Normalize components within each state to remove metro / rural bias
        """
        print("🔧 Applying state-specific normalization...")

        for col in self.feature_cols:
            if col not in self.components.columns:
                continue

            norm_col = f"{col}_state_norm"
            self.components[norm_col] = 0.0

            for state, group in self.components.groupby("state"):
                min_val = group[col].min()
                max_val = group[col].max()

                if max_val - min_val > 0:
                    self.components.loc[group.index, norm_col] = (
                        (group[col] - min_val) / (max_val - min_val)
                    ) * 100

                else:
                    self.components.loc[group.index, norm_col] = 0

        print("   ✅ State-level normalization complete")

    def apply_per_capita_weighting(self):
        """
        Weight UFI score by enrollment-based population proxy
        """
        print("🔧 Applying per-capita (enrollment-weighted) adjustment...")

        if "total_enrollments" not in self.components.columns:
            print("⚠️ total_enrollments missing — skipping per-capita weighting")
            return

        # Normalize enrollment weights
        weights = self.components["total_enrollments"]
        weights = weights / weights.max()

        self.components["UFI_weighted"] = self.components["UFI"] * weights
        print("   ✅ Per-capita weighting applied")


# USAGE EXAMPLE
if __name__ == "__main__":
    # Load UFI components
    components_df = pd.read_csv("data/processed/ufi_components.csv")

    # Initialize calculator
    calculator = UFICalculator(components_df)

    # Compute UFI with PCA weighting
    ufi_results = calculator.compute_ufi(weighting_method="pca")

    # Display top insights
    print("\n" + "=" * 60)
    print("🔥 TOP 10 HIGH FRICTION DISTRICTS")
    print("=" * 60)
    print(calculator.get_top_friction_districts(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("🌟 TOP 10 LOW FRICTION DISTRICTS")
    print("=" * 60)
    print(calculator.get_low_friction_districts(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("📊 STATE-LEVEL UFI SUMMARY")
    print("=" * 60)
    print(calculator.get_state_summary().head(15).to_string(index=False))

    # Export all results
    calculator.export_results()

    print("\n✅ UFI CALCULATION COMPLETE!")
