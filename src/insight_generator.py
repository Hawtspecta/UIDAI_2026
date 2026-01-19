"""
UIDAI HACKATHON 2026 - INSIGHT GENERATOR
Automated extraction of policy-relevant findings from UFI data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


class UFIInsightGenerator:
    """Generate actionable insights from UFI analysis"""

    def __init__(self, ufi_scores_path="data/processed/ufi_scores.csv"):
        self.ufi_data = pd.read_csv(ufi_scores_path)
        self.insights = {}

    def identify_update_deserts(self, threshold=10):
        """
        Find districts with high enrollment but very low update activity
        = Infrastructure access gaps
        """
        print("🔍 Identifying Update Deserts...")

        deserts = self.ufi_data[
            (
                self.ufi_data["total_enrollments"]
                > self.ufi_data["total_enrollments"].quantile(0.5)
            )
            & (self.ufi_data["UFI"] < threshold)
        ].sort_values("total_enrollments", ascending=False)

        self.insights["update_deserts"] = {
            "count": len(deserts),
            "top_10": deserts.head(10)[
                ["state", "district", "UFI", "total_enrollments"]
            ].to_dict("records"),
            "interpretation": f"Found {len(deserts)} districts with high population but minimal update activity. These likely face infrastructure or awareness gaps.",
            "recommendation": "Deploy mobile enrollment units and awareness campaigns in these districts.",
        }

        print(f"   ✅ Found {len(deserts)} Update Desert districts")
        return deserts

    def identify_high_friction_zones(self, threshold=75):
        """
        Find districts with very high UFI scores
        = System stress, migration hotspots, or policy impact areas
        """
        print("🔍 Identifying High Friction Zones...")

        high_friction = self.ufi_data[self.ufi_data["UFI"] > threshold].sort_values(
            "UFI", ascending=False
        )

        self.insights["high_friction_zones"] = {
            "count": len(high_friction),
            "top_10": high_friction.head(10)[
                ["state", "district", "UFI", "UFI_Category"]
            ].to_dict("records"),
            "interpretation": f"Found {len(high_friction)} districts experiencing severe system friction. These may indicate migration hotspots, rapid urbanization, or policy change impacts.",
            "recommendation": "Increase UIDAI center capacity and investigate root causes of high update demand.",
        }

        print(f"   ✅ Found {len(high_friction)} High Friction districts")
        return high_friction

    def analyze_digital_age_gap_leaders(self):
        """
        Identify districts with largest generational digital divide
        """
        print("🔍 Analyzing Digital Age Gap...")

        age_gap_leaders = self.ufi_data.nlargest(10, "age_disparity")[
            ["state", "district", "age_disparity", "UFI"]
        ]

        self.insights["digital_age_gap"] = {
            "top_10": age_gap_leaders.to_dict("records"),
            "max_gap": float(self.ufi_data["age_disparity"].max()),
            "mean_gap": float(self.ufi_data["age_disparity"].mean()),
            "interpretation": "These districts show severe intergenerational digital inequality. Elderly populations are significantly less engaged with Aadhaar updates.",
            "recommendation": "Launch targeted elderly outreach programs with simplified update processes.",
        }

        print(f"   ✅ Max age gap: {self.ufi_data['age_disparity'].max():.2f}")
        return age_gap_leaders

    def identify_biometric_refresh_leaders(self):
        """
        Districts with highest biometric update momentum
        = Security-conscious populations or aging demographics
        """
        print("🔍 Identifying Biometric Refresh Leaders...")

        bio_leaders = self.ufi_data.nlargest(10, "bio_refresh_rate")[
            ["state", "district", "bio_refresh_rate", "UFI"]
        ]

        self.insights["biometric_leaders"] = {
            "top_10": bio_leaders.to_dict("records"),
            "interpretation": "These districts show accelerating biometric update rates, indicating either aging populations requiring re-enrollment or heightened security awareness.",
            "recommendation": "Study these districts as models for biometric adoption best practices.",
        }

        print(f"   ✅ Identified {len(bio_leaders)} biometric leaders")
        return bio_leaders

    def identify_volatile_districts(self):
        """
        Districts with highest temporal volatility
        = Unstable update patterns, possible policy shocks or seasonal migration
        """
        print("🔍 Identifying Volatile Districts...")

        volatile = self.ufi_data.nlargest(10, "temporal_volatility")[
            ["state", "district", "temporal_volatility", "UFI"]
        ]

        self.insights["volatile_districts"] = {
            "top_10": volatile.to_dict("records"),
            "interpretation": "These districts show erratic update patterns, suggesting seasonal migration, policy shocks, or external events driving sudden demand.",
            "recommendation": "Implement flexible staffing models to handle demand spikes in these districts.",
        }

        print(f"   ✅ Identified {len(volatile)} volatile districts")
        return volatile

    def analyze_state_level_patterns(self):
        """
        State-level UFI aggregation and ranking
        """
        print("🔍 Analyzing State-Level Patterns...")

        state_summary = (
            self.ufi_data.groupby("state")
            .agg(
                {
                    "UFI": ["mean", "median", "std"],
                    "district": "count",
                    "total_enrollments": "sum",
                }
            )
            .round(2)
        )

        state_summary.columns = [
            "UFI_Mean",
            "UFI_Median",
            "UFI_StdDev",
            "Num_Districts",
            "Total_Enrollments",
        ]
        state_summary = state_summary.sort_values(
            "UFI_Mean", ascending=False
        ).reset_index()

        self.insights["state_ranking"] = {
            "top_5_friction": state_summary.head(5).to_dict("records"),
            "bottom_5_friction": state_summary.tail(5).to_dict("records"),
            "interpretation": "State-level friction patterns reveal regional policy effectiveness and infrastructure maturity.",
        }

        print(f"   ✅ Analyzed {len(state_summary)} states")
        return state_summary

    def calculate_impact_metrics(self):
        """
        Calculate overall system health metrics
        """
        print("🔍 Calculating Impact Metrics...")

        total_districts = len(self.ufi_data)
        high_friction_pct = (self.ufi_data["UFI"] > 75).sum() / total_districts * 100
        low_friction_pct = (self.ufi_data["UFI"] < 25).sum() / total_districts * 100

        self.insights["impact_metrics"] = {
            "total_districts_analyzed": int(total_districts),
            "mean_ufi": float(self.ufi_data["UFI"].mean()),
            "median_ufi": float(self.ufi_data["UFI"].median()),
            "high_friction_percentage": float(high_friction_pct),
            "low_friction_percentage": float(low_friction_pct),
            "total_enrollments": int(self.ufi_data["total_enrollments"].sum()),
            "districts_needing_intervention": int((self.ufi_data["UFI"] > 75).sum()),
            "stable_districts": int((self.ufi_data["UFI"].between(25, 75)).sum()),
        }

        print(
            f"   ✅ System health: {low_friction_pct:.1f}% low friction, {high_friction_pct:.1f}% high friction"
        )

    def generate_all_insights(self):
        """
        Run all insight generators and compile report
        """
        print("\n" + "=" * 60)
        print("🚀 GENERATING UFI INSIGHTS")
        print("=" * 60 + "\n")

        self.identify_update_deserts()
        self.identify_high_friction_zones()
        self.analyze_digital_age_gap_leaders()
        self.identify_biometric_refresh_leaders()
        self.identify_volatile_districts()
        self.analyze_state_level_patterns()
        self.calculate_impact_metrics()

        print("\n" + "=" * 60)
        print("✅ INSIGHT GENERATION COMPLETE")
        print("=" * 60)

        return self.insights

    def export_insights(self, output_dir="outputs/reports"):
        """
        Export insights to JSON and text files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # JSON export
        json_path = f"{output_dir}/ufi_insights.json"
        with open(json_path, "w") as f:
            json.dump(self.insights, f, indent=2, default=str)
        print(f"💾 Saved insights to: {json_path}")

        # Human-readable text report
        txt_path = f"{output_dir}/ufi_insights_report.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("UIDAI UPDATE FRICTION INDEX (UFI) - KEY INSIGHTS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 70 + "\n")
            metrics = self.insights.get("impact_metrics", {})
            f.write(
                f"Total Districts Analyzed: {metrics.get('total_districts_analyzed', 'N/A')}\n"
            )
            f.write(f"Mean UFI Score: {metrics.get('mean_ufi', 'N/A'):.2f}\n")
            f.write(
                f"Districts Needing Intervention: {metrics.get('districts_needing_intervention', 'N/A')}\n\n"
            )

            f.write("KEY FINDINGS\n")
            f.write("-" * 70 + "\n\n")

            for category, data in self.insights.items():
                if category == "impact_metrics":
                    continue
                f.write(f"{category.upper().replace('_', ' ')}\n")
                f.write(f"Interpretation: {data.get('interpretation', 'N/A')}\n")
                f.write(f"Recommendation: {data.get('recommendation', 'N/A')}\n\n")

        print(f"💾 Saved report to: {txt_path}")

        return json_path, txt_path


if __name__ == "__main__":
    generator = UFIInsightGenerator()
    insights = generator.generate_all_insights()
    generator.export_insights()

    print("\n" + "=" * 60)
    print("📊 TOP ACTIONABLE INSIGHTS:")
    print("=" * 60)

    print(
        f"\n1. UPDATE DESERTS: {insights['update_deserts']['count']} districts need infrastructure support"
    )
    print(
        f"2. HIGH FRICTION ZONES: {insights['high_friction_zones']['count']} districts experiencing severe stress"
    )
    print(
        f"3. SYSTEM HEALTH: {insights['impact_metrics']['low_friction_percentage']:.1f}% districts stable"
    )
    print(
        f"4. INTERVENTION NEEDED: {insights['impact_metrics']['districts_needing_intervention']} districts"
    )
