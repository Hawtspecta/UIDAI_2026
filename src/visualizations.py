"""
UIDAI HACKATHON 2026 - VISUALIZATION ENGINE
Generate high-quality charts for PDF submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class UFIVisualizer:
    """Generate all visualizations for UFI analysis"""
    
    def __init__(self, ufi_scores_path='data/processed/ufi_scores.csv'):
        self.ufi_data = pd.read_csv(ufi_scores_path)
        self.output_dir = 'outputs/figures'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_ufi_distribution(self):
        """UFI score distribution histogram"""
        print("📊 Creating UFI distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(self.ufi_data['UFI'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.ufi_data['UFI'].mean(), color='red', linestyle='--', label=f'Mean: {self.ufi_data["UFI"].mean():.2f}')
        ax1.axvline(self.ufi_data['UFI'].median(), color='green', linestyle='--', label=f'Median: {self.ufi_data["UFI"].median():.2f}')
        ax1.set_xlabel('UFI Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Update Friction Index Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot by category
        category_order = ['Low Friction', 'Moderate Friction', 'High Friction', 'Very High Friction']
        self.ufi_data['UFI_Category'] = pd.Categorical(self.ufi_data['UFI_Category'], categories=category_order, ordered=True)
        
        sns.boxplot(data=self.ufi_data, x='UFI_Category', y='UFI', ax=ax2, palette='RdYlGn_r')
        ax2.set_xlabel('UFI Category')
        ax2.set_ylabel('UFI Score')
        ax2.set_title('UFI Distribution by Category')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/ufi_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def plot_component_correlation(self):
        """Correlation heatmap of UFI components"""
        print("📊 Creating component correlation heatmap...")
        
        component_cols = [
            'demo_update_intensity',
            'bio_refresh_rate',
            'age_disparity',
            'update_enrol_ratio',
            'temporal_volatility'
        ]
        
        corr_matrix = self.ufi_data[component_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('UFI Component Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/component_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def plot_top_states_ufi(self, top_n=15):
        """Bar chart of states by mean UFI"""
        print(f"📊 Creating top {top_n} states UFI chart...")
        
        state_ufi = self.ufi_data.groupby('state')['UFI'].mean().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(state_ufi)), state_ufi.values, color='coral', edgecolor='black')
        ax.set_yticks(range(len(state_ufi)))
        ax.set_yticklabels(state_ufi.index)
        ax.set_xlabel('Mean UFI Score', fontweight='bold')
        ax.set_title(f'Top {top_n} States by Update Friction Index', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, val) in enumerate(state_ufi.items()):
            ax.text(val + 1, i, f'{val:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/top_states_ufi.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def plot_component_contributions(self):
        """Stacked bar showing component contributions to UFI"""
        print("📊 Creating component contribution analysis...")
        
        # Get top 10 high friction districts
        top_10 = self.ufi_data.nlargest(10, 'UFI')
        
        component_cols = [
            'demo_update_intensity_normalized',
            'bio_refresh_rate_normalized',
            'age_disparity_normalized',
            'update_enrol_ratio_normalized',
            'temporal_volatility_normalized'
        ]
        
        # Check which normalized columns exist
        available_cols = [col for col in component_cols if col in top_10.columns]
        
        if not available_cols:
            print("   ⚠️  Normalized columns not found, skipping this visualization")
            return None
        
        plot_data = top_10[['state', 'district'] + available_cols].copy()
        plot_data['label'] = plot_data['state'] + ' - ' + plot_data['district']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        plot_data.set_index('label')[available_cols].plot(
            kind='barh',
            stacked=True,
            ax=ax,
            colormap='tab10'
        )
        
        ax.set_xlabel('Normalized Component Score', fontweight='bold')
        ax.set_title('UFI Component Breakdown - Top 10 High Friction Districts', fontsize=14, fontweight='bold')
        ax.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/component_contributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def plot_scatter_matrix(self):
        """Scatter plot: UFI vs key components"""
        print("📊 Creating UFI vs components scatter plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        components = [
            ('demo_update_intensity', 'Demographic Update Intensity'),
            ('age_disparity', 'Age Group Disparity'),
            ('update_enrol_ratio', 'Update-Enrollment Ratio'),
            ('temporal_volatility', 'Temporal Volatility')
        ]
        
        for ax, (col, title) in zip(axes.flat, components):
            ax.scatter(self.ufi_data[col], self.ufi_data['UFI'], alpha=0.5, s=30, color='steelblue')
            ax.set_xlabel(title, fontweight='bold')
            ax.set_ylabel('UFI Score', fontweight='bold')
            ax.set_title(f'UFI vs {title}')
            ax.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(self.ufi_data[col], self.ufi_data['UFI'], 1)
            p = np.poly1d(z)
            ax.plot(self.ufi_data[col], p(self.ufi_data[col]), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/ufi_component_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def plot_category_distribution(self):
        """Pie chart of UFI category distribution"""
        print("📊 Creating UFI category distribution pie chart...")
        
        category_counts = self.ufi_data['UFI_Category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        
        wedges, texts, autotexts = ax.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('District Distribution by UFI Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/ufi_category_pie.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to {output_path}")
        return output_path
    
    def create_interactive_state_map(self):
        """Interactive Plotly choropleth map (if possible)"""
        print("📊 Creating interactive state UFI map...")
        
        state_ufi = self.ufi_data.groupby('state')['UFI'].mean().reset_index()
        
        fig = px.bar(
            state_ufi.sort_values('UFI', ascending=False),
            x='UFI',
            y='state',
            orientation='h',
            title='State-Level Update Friction Index',
            color='UFI',
            color_continuous_scale='RdYlGn_r',
            labels={'UFI': 'Mean UFI Score', 'state': 'State'}
        )
        
        fig.update_layout(height=800, showlegend=False)
        
        output_path = f'{self.output_dir}/interactive_state_ufi.html'
        fig.write_html(output_path)
        
        print(f"   ✅ Saved interactive map to {output_path}")
        return output_path
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("🎨 GENERATING ALL VISUALIZATIONS")
        print("="*60 + "\n")
        
        outputs = {}
        
        outputs['distribution'] = self.plot_ufi_distribution()
        outputs['correlation'] = self.plot_component_correlation()
        outputs['top_states'] = self.plot_top_states_ufi()
        outputs['contributions'] = self.plot_component_contributions()
        outputs['scatter'] = self.plot_scatter_matrix()
        outputs['category_pie'] = self.plot_category_distribution()
        outputs['interactive_map'] = self.create_interactive_state_map()
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS GENERATED")
        print("="*60)
        
        return outputs


if __name__ == "__main__":
    visualizer = UFIVisualizer()
    outputs = visualizer.generate_all_visualizations()
    
    print("\n📊 VISUALIZATION SUMMARY:")
    for name, path in outputs.items():
        if path:
            print(f"   ✅ {name}: {path}")