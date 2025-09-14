"""
Sophisticated AI Dashboard Agent - Creates intelligent visualizations based on data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
from .base_agent import BaseAgent
from typing import List, Dict, Any
import warnings
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class DashboardAgent(BaseAgent):
    """Sophisticated AI agent that creates intelligent dashboards based on data analysis"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Sophisticated AI Dashboard Agent"
        self.chart_styles = {
            'executive': {'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 'style': 'professional'},
            'creative': {'colors': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'], 'style': 'modern'},
            'analytical': {'colors': ['#2E8B57', '#FFD700', '#FF6347', '#4169E1', '#32CD32'], 'style': 'scientific'}
        }
    
    def create_sophisticated_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a sophisticated dashboard with AI-driven insights"""
        try:
            # Analyze data comprehensively
            analysis = self._comprehensive_data_analysis(df)
            
            # Generate intelligent insights
            insights = self._generate_ai_insights(df, analysis)
            
            # Create sophisticated visualizations
            charts = self._create_intelligent_visualizations(df, analysis)
            
            # Calculate executive metrics
            kpis = self._calculate_executive_kpis(df, analysis)
            
            return {
                'charts': charts,
                'executive_metrics': kpis,
                'ai_insights': insights,
                'analysis': analysis,
                'dashboard_type': 'sophisticated_ai_generated',
                'creation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._create_error_dashboard(str(e))
    
    def _comprehensive_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analysis of the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Advanced statistical analysis
        correlations = {}
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Find strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        correlations[f"{corr_matrix.columns[i]}_vs_{corr_matrix.columns[j]}"] = corr_val
        
        # Detect business domain
        domain = self._detect_business_domain(df)
        
        # Identify key patterns
        patterns = self._identify_data_patterns(df, numeric_cols, categorical_cols)
        
        return {
            'shape': df.shape,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'correlations': correlations,
            'business_domain': domain,
            'patterns': patterns,
            'data_quality': self._assess_data_quality(df),
            'statistical_summary': self._get_statistical_summary(df, numeric_cols)
        }
    
    def _detect_business_domain(self, df: pd.DataFrame) -> str:
        """Detect the business domain from column names and data patterns"""
        column_names = [col.lower() for col in df.columns]
        
        domain_indicators = {
            'sales': ['revenue', 'sales', 'profit', 'customer', 'deal', 'commission'],
            'finance': ['amount', 'balance', 'investment', 'portfolio', 'return', 'cost'],
            'marketing': ['campaign', 'conversion', 'click', 'impression', 'lead', 'ctr'],
            'ecommerce': ['order', 'product', 'cart', 'purchase', 'quantity', 'price'],
            'hr': ['employee', 'salary', 'department', 'performance', 'training'],
            'operations': ['production', 'inventory', 'quality', 'efficiency', 'supply']
        }
        
        domain_scores = {}
        for domain, keywords in domain_indicators.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in column_names))
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def _identify_data_patterns(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Identify interesting patterns in the data"""
        patterns = {
            'trends': [],
            'outliers': {},
            'distributions': {},
            'relationships': []
        }
        
        # Analyze trends in numeric data
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 5:
                # Simple trend detection
                x = np.arange(len(data))
                slope = np.polyfit(x, data.values, 1)[0]
                if abs(slope) > data.std() * 0.1:
                    direction = 'increasing' if slope > 0 else 'decreasing'
                    patterns['trends'].append({'column': col, 'direction': direction, 'strength': abs(slope)})
        
        # Detect outliers
        for col in numeric_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                patterns['outliers'][col] = len(outliers)
        
        return patterns  
  
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the dataset"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        return {
            'completeness': ((total_cells - missing_cells) / total_cells) * 100,
            'missing_values': missing_cells,
            'duplicate_rows': df.duplicated().sum(),
            'total_records': len(df),
            'total_columns': len(df.columns)
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Get statistical summary of numeric columns"""
        if not numeric_cols:
            return {}
        
        summary = {}
        for col in numeric_cols:
            data = df[col].dropna()
            summary[col] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            }
        
        return summary
    
    def _generate_ai_insights(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate AI-powered insights about the data"""
        domain = analysis['business_domain']
        patterns = analysis['patterns']
        quality = analysis['data_quality']
        
        insights = []
        
        # Data overview insight
        insights.append(f"📊 **Data Overview**: Analyzed {len(df):,} records across {len(df.columns)} dimensions with {quality['completeness']:.1f}% data completeness.")
        
        # Domain-specific insight
        insights.append(f"🏢 **Business Context**: This appears to be {domain} data based on column patterns and content analysis.")
        
        # Pattern insights
        if patterns['trends']:
            trend_cols = [t['column'] for t in patterns['trends']]
            insights.append(f"📈 **Trend Analysis**: Detected significant trends in {', '.join(trend_cols[:3])}.")
        
        # Correlation insights
        if analysis['correlations']:
            strong_corrs = len(analysis['correlations'])
            insights.append(f"🔗 **Relationships**: Found {strong_corrs} strong correlations between variables.")
        
        # Quality insights
        if quality['duplicate_rows'] > 0:
            insights.append(f"⚠️ **Data Quality**: {quality['duplicate_rows']} duplicate records detected - consider data cleaning.")
        
        # Outlier insights
        outlier_cols = list(analysis['patterns']['outliers'].keys())
        if outlier_cols:
            insights.append(f"🎯 **Anomalies**: Outliers detected in {', '.join(outlier_cols[:3])} - potential areas for investigation.")
        
        return "\n\n".join(insights)
    
    def _create_intelligent_visualizations(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create intelligent visualizations based on data analysis"""
        charts = []
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        domain = analysis['business_domain']
        
        try:
            # Chart 1: Executive Summary Dashboard
            fig1 = self._create_executive_summary_chart(df, analysis)
            if fig1:
                reasoning = self._explain_chart_selection('executive_summary', df, analysis)
                detailed_explanation = self._generate_detailed_chart_explanation('executive_summary', df, analysis)
                charts.append({
                    'title': f'🚀 {domain.title()} Executive Dashboard',
                    'chart': fig1,
                    'type': 'executive_summary',
                    'description': 'Comprehensive overview of key business metrics and performance indicators',
                    'ai_reasoning': reasoning,
                    'detailed_explanation': detailed_explanation
                })
            
            # Chart 2: Advanced Analytics Chart
            if len(numeric_cols) >= 2:
                fig2 = self._create_advanced_analytics_chart(df, analysis)
                if fig2:
                    reasoning = self._explain_chart_selection('advanced_analytics', df, analysis)
                    detailed_explanation = self._generate_detailed_chart_explanation('advanced_analytics', df, analysis)
                    charts.append({
                        'title': '🔬 Advanced Analytics & Correlations',
                        'chart': fig2,
                        'type': 'advanced_analytics',
                        'description': 'Deep dive into data relationships and statistical patterns',
                        'ai_reasoning': reasoning,
                        'detailed_explanation': detailed_explanation
                    })
            
            # Chart 3: Performance Intelligence
            if categorical_cols and numeric_cols:
                fig3 = self._create_performance_intelligence_chart(df, analysis)
                if fig3:
                    reasoning = self._explain_chart_selection('performance_intelligence', df, analysis)
                    detailed_explanation = self._generate_detailed_chart_explanation('performance_intelligence', df, analysis)
                    charts.append({
                        'title': '🎯 Performance Intelligence',
                        'chart': fig3,
                        'type': 'performance_intelligence',
                        'description': 'Category-wise performance analysis with benchmarking',
                        'ai_reasoning': reasoning,
                        'detailed_explanation': detailed_explanation
                    })
            
            # Chart 4: Predictive Insights
            if len(numeric_cols) > 0:
                fig4 = self._create_predictive_insights_chart(df, analysis)
                if fig4:
                    reasoning = self._explain_chart_selection('predictive_insights', df, analysis)
                    detailed_explanation = self._generate_detailed_chart_explanation('predictive_insights', df, analysis)
                    charts.append({
                        'title': '🔮 Predictive Insights & Forecasting',
                        'chart': fig4,
                        'type': 'predictive_insights',
                        'description': 'Statistical forecasting and trend projection analysis',
                        'ai_reasoning': reasoning,
                        'detailed_explanation': detailed_explanation
                    })
            
        except Exception as e:
            # Fallback chart
            fig_fallback = self._create_fallback_chart(df)
            charts.append({
                'title': '📊 Data Overview',
                'chart': fig_fallback,
                'type': 'fallback',
                'description': 'Basic data visualization and summary',
                'ai_reasoning': 'Fallback visualization created due to data complexity or processing constraints.',
                'detailed_explanation': 'This chart provides a basic overview of your data structure and key statistics.'
            })
        
        return charts
    
    def _explain_chart_selection(self, chart_type: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Explain why this specific chart type was selected for the data"""
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        domain = analysis['business_domain']
        correlations = analysis['correlations']
        patterns = analysis['patterns']
        
        explanations = {
            'executive_summary': f"""
🧠 **AI Chart Selection Reasoning:**
I chose an Executive Summary Dashboard because:
• **Multi-dimensional Analysis**: Your data has {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables requiring comprehensive overview
• **Business Context**: Detected {domain} domain data which benefits from executive-level KPI visualization
• **Pattern Complexity**: Found {len(correlations)} significant correlations and {len(patterns['trends'])} trends requiring integrated view
• **Decision Support**: Executive dashboards provide the holistic view needed for strategic decision-making
            """,
            
            'advanced_analytics': f"""
🧠 **AI Chart Selection Reasoning:**
I selected Advanced Analytics visualization because:
• **Statistical Depth**: {len(numeric_cols)} numeric variables enable sophisticated correlation and distribution analysis
• **Relationship Discovery**: Detected {len(correlations)} strong correlations (>0.5) requiring network visualization
• **Outlier Insights**: Found outliers in {len(patterns['outliers'])} variables needing box plot analysis
• **Predictive Value**: Statistical patterns suggest regression analysis will reveal actionable insights
            """,
            
            'performance_intelligence': f"""
🧠 **AI Chart Selection Reasoning:**
I chose Performance Intelligence charts because:
• **Category Analysis**: {len(categorical_cols)} categorical variables enable performance comparison across segments
• **Benchmarking Need**: Business data requires ranking and performance matrix analysis
• **Volume vs Quality**: Data structure supports bubble charts showing performance vs volume relationships
• **Distribution Insights**: Violin plots reveal performance consistency patterns across categories
            """,
            
            'predictive_insights': f"""
🧠 **AI Chart Selection Reasoning:**
I selected Predictive Insights visualization because:
• **Trend Detection**: Identified {len(patterns['trends'])} significant trends requiring forecasting analysis
• **Time Series Potential**: Data sequence suggests temporal patterns worth projecting
• **Risk Assessment**: Statistical analysis enables confidence interval and scenario modeling
• **Strategic Planning**: Forecasting supports forward-looking business decision making
            """
        }
        
        return explanations.get(chart_type, "Chart selected based on data characteristics and business context.")
    
    def _generate_detailed_chart_explanation(self, chart_type: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate detailed explanation of what each chart shows and why it's valuable"""
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        domain = analysis['business_domain']
        quality = analysis['data_quality']
        
        explanations = {
            'executive_summary': f"""
📊 **Detailed Chart Explanation:**

**What This Dashboard Shows:**
• **Key Metrics Panel**: Displays average values of your {len(numeric_cols)} primary performance indicators
• **Trend Analysis**: Shows the trajectory of your main metric ({numeric_cols[0] if numeric_cols else 'primary metric'}) with predictive trend line
• **Distribution Analysis**: Reveals the spread and central tendencies of your data with statistical markers
• **Category Performance**: Breaks down performance by {categorical_cols[0] if categorical_cols else 'categories'} showing relative contributions
• **Correlation Matrix**: Maps relationships between all numeric variables to identify connected metrics

**Business Value:**
• **Strategic Overview**: Get a complete picture of your {domain} performance at a glance
• **Pattern Recognition**: Quickly identify which metrics move together and influence each other
• **Performance Benchmarking**: See which categories are over/under-performing relative to others
• **Data Quality Insight**: Understand your data completeness ({quality['completeness']:.1f}%) and reliability

**Key Insights to Look For:**
• Strong correlations (red/blue in heatmap) indicate metrics that influence each other
• Trend direction shows whether your business is growing or declining
• Distribution shape reveals if performance is consistent or highly variable
• Category sizes in pie chart show where to focus improvement efforts
            """,
            
            'advanced_analytics': f"""
🔬 **Detailed Chart Explanation:**

**What This Analysis Shows:**
• **Correlation Network**: Visual map of how your {len(numeric_cols)} variables influence each other
• **Distribution Comparison**: Side-by-side comparison of data spread patterns
• **Outlier Analysis**: Box plots revealing unusual values that need investigation
• **Relationship Analysis**: Scatter plot with regression showing predictive relationships

**Statistical Insights:**
• **Network Connections**: Thick lines = strong relationships, colors indicate positive (blue) vs negative (red) correlations
• **Distribution Shapes**: Normal curves suggest stable processes, skewed distributions indicate opportunities
• **Box Plot Whiskers**: Points outside whiskers are outliers requiring attention
• **Regression Line**: Shows predictive relationship strength (R² value indicates reliability)

**Business Applications:**
• **Root Cause Analysis**: Use correlation network to trace performance drivers
• **Quality Control**: Outliers may indicate process issues or exceptional performance
• **Predictive Modeling**: Strong relationships enable forecasting one metric from another
• **Process Optimization**: Distribution patterns reveal consistency and improvement opportunities

**Action Items:**
• Investigate outliers for process improvements or exceptional cases
• Leverage strong correlations for predictive analytics
• Address distribution skewness for more consistent performance
            """,
            
            'performance_intelligence': f"""
🎯 **Detailed Chart Explanation:**

**What This Intelligence Shows:**
• **Performance Ranking**: Horizontal bar chart ranking {categorical_cols[0] if categorical_cols else 'categories'} by average performance
• **Performance vs Volume**: Bubble chart showing relationship between activity level and results
• **Performance Distribution**: Violin plots revealing consistency patterns across categories
• **Performance Matrix**: Scatter plot mapping performance level against variability

**Performance Insights:**
• **Ranking Order**: Top performers are your benchmark categories to study and replicate
• **Bubble Size**: Larger bubbles represent higher total impact/volume
• **Violin Width**: Wider sections show common performance levels, narrow sections show rare outcomes
• **Matrix Quadrants**: Top-right = high performance + low variability (ideal), bottom-left = needs improvement

**Strategic Applications:**
• **Best Practice Identification**: Study top-ranked categories for success factors
• **Resource Allocation**: Focus on high-volume, high-performance opportunities
• **Risk Management**: Address high-variability categories for more predictable results
• **Benchmarking**: Use performance matrix to set realistic improvement targets

**Recommended Actions:**
• Replicate strategies from top-performing categories
• Investigate why some categories have high variability
• Focus improvement efforts on high-volume, low-performance areas
            """,
            
            'predictive_insights': f"""
🔮 **Detailed Chart Explanation:**

**What This Forecasting Shows:**
• **Trend Forecasting**: Historical data with projected future values and confidence bands
• **Seasonal Patterns**: Cyclical patterns that repeat over time periods
• **Growth Analysis**: Rate of change analysis showing acceleration or deceleration
• **Scenario Modeling**: Multiple potential futures based on different assumptions

**Predictive Elements:**
• **Trend Line**: Mathematical projection based on historical patterns
• **Confidence Bands**: Range of likely outcomes (wider = more uncertainty)
• **Seasonal Components**: Recurring patterns that affect predictions
• **Growth Rates**: Percentage change rates for planning purposes

**Business Planning Value:**
• **Budget Forecasting**: Project future resource needs based on trends
• **Capacity Planning**: Anticipate when current systems will reach limits
• **Risk Assessment**: Understand uncertainty ranges for contingency planning
• **Goal Setting**: Set realistic targets based on historical performance patterns

**Planning Applications:**
• Use trend projections for annual budget planning
• Monitor actual vs predicted values to refine forecasting accuracy
• Plan capacity expansions before hitting projected limits
• Develop contingency plans for confidence band extremes
            """
        }
        
        return explanations.get(chart_type, "This chart provides insights into your data patterns and business performance.")
 
    def _create_executive_summary_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> plt.Figure:
        """Create an executive summary dashboard with multiple panels"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Set professional styling
        try:
            plt.style.use('default')
        except:
            pass  # Use current style if default fails
        colors = self.chart_styles['executive']['colors']
        
        # Main title
        fig.suptitle('🚀 Executive Business Intelligence Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        
        # Panel 1: Key Metrics Overview (Top Left, 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if numeric_cols:
            metrics = df[numeric_cols].mean()
            bars = ax1.bar(range(len(metrics)), metrics.values, color=colors[:len(metrics)])
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels([col.replace('_', '\n').title() for col in metrics.index], rotation=0)
            ax1.set_title('📊 Key Performance Metrics', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, metrics.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: Trend Analysis (Top Right, 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        if numeric_cols:
            primary_col = numeric_cols[0]
            data = df[primary_col].dropna()
            x = range(len(data))
            ax2.plot(x, data.values, color=colors[0], linewidth=3, marker='o', markersize=4)
            ax2.fill_between(x, data.values, alpha=0.3, color=colors[0])
            
            # Add trend line
            z = np.polyfit(x, data.values, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "--", color=colors[1], linewidth=2, alpha=0.8)
            
            ax2.set_title(f'📈 {primary_col.title()} Trend Analysis', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Distribution Analysis (Middle Left, 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        if numeric_cols:
            col = numeric_cols[0]
            data = df[col].dropna()
            n, bins, patches = ax3.hist(data, bins=25, alpha=0.8, color=colors[1], edgecolor='white')
            
            # Color gradient
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.viridis(i / len(patches)))
            
            # Statistical lines
            mean_val = data.mean()
            median_val = data.median()
            ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax3.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax3.set_title(f'📊 {col.title()} Distribution', fontsize=14, fontweight='bold')
            ax3.legend()
        
        # Panel 4: Category Performance (Middle Right, 2 columns)
        ax4 = fig.add_subplot(gs[1, 2:])
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            category_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(8)
            
            wedges, texts, autotexts = ax4.pie(category_data.values, labels=category_data.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title(f'🎯 {cat_col.title()} Performance', fontsize=14, fontweight='bold')
        
        # Panel 5: Correlation Heatmap (Bottom, full width)
        ax5 = fig.add_subplot(gs[2, :])
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            im = ax5.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if not mask[i, j]:
                        text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
            
            ax5.set_xticks(range(len(corr_matrix.columns)))
            ax5.set_yticks(range(len(corr_matrix.columns)))
            ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax5.set_yticklabels(corr_matrix.columns)
            ax5.set_title('🔗 Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig 
   
    def _create_advanced_analytics_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> plt.Figure:
        """Create advanced analytics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔬 Advanced Analytics & Statistical Intelligence', fontsize=18, fontweight='bold')
        
        numeric_cols = analysis['numeric_columns']
        colors = self.chart_styles['analytical']['colors']
        
        # Chart 1: Correlation Network
        ax1.set_title('🕸️ Correlation Network', fontweight='bold')
        if len(numeric_cols) >= 3:
            corr_matrix = df[numeric_cols].corr()
            
            # Create network layout
            n_vars = min(len(numeric_cols), 8)  # Limit for readability
            angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            # Draw connections for strong correlations
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if i < len(corr_matrix.columns) and j < len(corr_matrix.columns):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:
                            line_width = abs(corr_val) * 5
                            line_color = colors[0] if corr_val > 0 else colors[3]
                            alpha = abs(corr_val) * 0.8
                            ax1.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                                   color=line_color, linewidth=line_width, alpha=alpha)
            
            # Draw nodes
            for i in range(n_vars):
                if i < len(numeric_cols):
                    ax1.scatter(x_pos[i], y_pos[i], s=500, c=colors[1], alpha=0.8, edgecolors='white', linewidth=2)
                    ax1.text(x_pos[i], y_pos[i], numeric_cols[i][:8], ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
            
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.axis('off')
        
        # Chart 2: Statistical Distribution Comparison
        ax2.set_title('📊 Distribution Comparison', fontweight='bold')
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            ax2.hist(df[col1].dropna(), alpha=0.7, color=colors[0], label=col1, bins=20)
            ax2.hist(df[col2].dropna(), alpha=0.7, color=colors[1], label=col2, bins=20)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Box Plot Analysis
        ax3.set_title('📦 Outlier Analysis', fontweight='bold')
        if numeric_cols:
            box_data = [df[col].dropna() for col in numeric_cols[:5]]
            bp = ax3.boxplot(box_data, labels=[col[:8] for col in numeric_cols[:5]], patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Chart 4: Scatter Plot with Regression
        ax4.set_title('💫 Relationship Analysis', fontweight='bold')
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            ax4.scatter(df[col1], df[col2], alpha=0.6, color=colors[2], s=50)
            
            # Add regression line
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data[col1], valid_data[col2], 1)
                p = np.poly1d(z)
                ax4.plot(valid_data[col1], p(valid_data[col1]), "r--", alpha=0.8, linewidth=2)
                
                # Calculate R-squared
                correlation = valid_data[col1].corr(valid_data[col2])
                ax4.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax4.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax4.set_xlabel(col1)
            ax4.set_ylabel(col2)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_performance_intelligence_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> plt.Figure:
        """Create performance intelligence visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎯 Performance Intelligence & Benchmarking', fontsize=18, fontweight='bold')
        
        categorical_cols = analysis['categorical_columns']
        numeric_cols = analysis['numeric_columns']
        colors = self.chart_styles['creative']['colors']
        
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Group data
            grouped = df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count', 'std']).reset_index()
            grouped = grouped.sort_values('mean', ascending=False).head(10)
            
            # Chart 1: Performance Ranking
            ax1.set_title('🏆 Performance Ranking', fontweight='bold')
            bars = ax1.barh(range(len(grouped)), grouped['mean'], color=colors)
            ax1.set_yticks(range(len(grouped)))
            ax1.set_yticklabels(grouped[cat_col])
            ax1.set_xlabel(f'Average {num_col.title()}')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, grouped['mean'])):
                ax1.text(bar.get_width() + max(grouped['mean'])*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}', va='center', fontweight='bold')
            
            # Chart 2: Performance vs Volume Bubble Chart
            ax2.set_title('🎈 Performance vs Volume', fontweight='bold')
            scatter = ax2.scatter(grouped['count'], grouped['mean'], s=grouped['sum']/grouped['sum'].max()*500, 
                                c=grouped['mean'], cmap='viridis', alpha=0.7, edgecolors='white', linewidth=2)
            ax2.set_xlabel('Volume (Count)')
            ax2.set_ylabel(f'Performance ({num_col.title()})')
            
            # Add category labels
            for i, row in grouped.iterrows():
                ax2.annotate(row[cat_col][:8], (row['count'], row['mean']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Chart 3: Performance Distribution
            ax3.set_title('📊 Performance Distribution', fontweight='bold')
            categories = grouped[cat_col].head(6)
            violin_data = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
            
            if all(len(data) > 0 for data in violin_data):
                parts = ax3.violinplot(violin_data, positions=range(len(categories)), showmeans=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[0])
                    pc.set_alpha(0.7)
                
                ax3.set_xticks(range(len(categories)))
                ax3.set_xticklabels([cat[:8] for cat in categories], rotation=45)
                ax3.set_ylabel(num_col.title())
            
            # Chart 4: Performance Matrix
            ax4.set_title('🎯 Performance Matrix', fontweight='bold')
            if len(grouped) >= 4:
                # Create performance vs consistency matrix
                ax4.scatter(grouped['mean'], grouped['std'], s=grouped['count']*5, 
                          c=colors[2], alpha=0.7, edgecolors='white', linewidth=2)
                
                # Add quadrant lines
                mean_perf = grouped['mean'].median()
                mean_std = grouped['std'].median()
                ax4.axvline(mean_perf, color='gray', linestyle='--', alpha=0.5)
                ax4.axhline(mean_std, color='gray', linestyle='--', alpha=0.5)
                
                ax4.set_xlabel('Performance Level')
                ax4.set_ylabel('Variability (Std Dev)')
                
                # Add quadrant labels
                ax4.text(0.75, 0.95, 'High Perf\nHigh Var', transform=ax4.transAxes, ha='center', va='top')
                ax4.text(0.25, 0.95, 'Low Perf\nHigh Var', transform=ax4.transAxes, ha='center', va='top')
                ax4.text(0.75, 0.05, 'High Perf\nLow Var', transform=ax4.transAxes, ha='center', va='bottom')
                ax4.text(0.25, 0.05, 'Low Perf\nLow Var', transform=ax4.transAxes, ha='center', va='bottom')
        
        plt.tight_layout()
        return fig    

    def _create_predictive_insights_chart(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> plt.Figure:
        """Create predictive insights and forecasting visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔮 Predictive Insights & Forecasting Intelligence', fontsize=18, fontweight='bold')
        
        numeric_cols = analysis['numeric_columns']
        colors = self.chart_styles['executive']['colors']
        
        if numeric_cols:
            primary_col = numeric_cols[0]
            data = df[primary_col].dropna()
            
            # Chart 1: Trend Forecasting
            ax1.set_title('📈 Trend Forecasting', fontweight='bold')
            x = np.arange(len(data))
            ax1.plot(x, data.values, color=colors[0], linewidth=3, label='Historical Data', marker='o', markersize=4)
            
            # Simple linear forecast
            if len(data) > 2:
                z = np.polyfit(x, data.values, 1)
                p = np.poly1d(z)
                
                # Extend forecast
                forecast_x = np.arange(len(data), len(data) + 10)
                forecast_y = p(forecast_x)
                
                ax1.plot(np.concatenate([x, forecast_x]), np.concatenate([p(x), forecast_y]), 
                        '--', color=colors[1], linewidth=2, label='Trend Forecast', alpha=0.8)
                ax1.fill_between(forecast_x, forecast_y * 0.9, forecast_y * 1.1, 
                               alpha=0.3, color=colors[1], label='Confidence Band')
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('Time Period')
            ax1.set_ylabel(primary_col.title())
            
            # Chart 2: Statistical Control Chart
            ax2.set_title('📊 Statistical Control Chart', fontweight='bold')
            mean_val = data.mean()
            std_val = data.std()
            
            ax2.plot(x, data.values, color=colors[0], linewidth=2, marker='o', markersize=3)
            ax2.axhline(mean_val, color=colors[1], linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax2.axhline(mean_val + 2*std_val, color=colors[3], linestyle='--', alpha=0.7, label='Upper Control')
            ax2.axhline(mean_val - 2*std_val, color=colors[3], linestyle='--', alpha=0.7, label='Lower Control')
            ax2.fill_between(x, mean_val - 2*std_val, mean_val + 2*std_val, alpha=0.2, color=colors[1])
            
            # Highlight out-of-control points
            out_of_control = (data > mean_val + 2*std_val) | (data < mean_val - 2*std_val)
            if out_of_control.any():
                ax2.scatter(x[out_of_control], data[out_of_control], color='red', s=100, 
                          marker='x', linewidth=3, label='Out of Control')
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Moving Statistics
            ax3.set_title('📈 Moving Statistics', fontweight='bold')
            if len(data) > 10:
                window = min(7, len(data) // 3)
                rolling_mean = data.rolling(window=window).mean()
                rolling_std = data.rolling(window=window).std()
                
                ax3.plot(x, rolling_mean, color=colors[0], linewidth=2, label=f'{window}-Period Mean')
                ax3.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                               alpha=0.3, color=colors[0], label='±1 Std Dev')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Chart 4: Seasonality & Patterns
            ax4.set_title('🔄 Pattern Analysis', fontweight='bold')
            if len(data) >= 12:
                # Simple seasonality detection
                period = min(12, len(data) // 2)
                seasonal_data = []
                for i in range(period):
                    seasonal_values = data[i::period]
                    if len(seasonal_values) > 0:
                        seasonal_data.append(seasonal_values.mean())
                
                if seasonal_data:
                    ax4.bar(range(len(seasonal_data)), seasonal_data, color=colors[2], alpha=0.7)
                    ax4.set_xlabel('Seasonal Period')
                    ax4.set_ylabel('Average Value')
                    ax4.grid(True, alpha=0.3)
            else:
                # Show distribution instead
                ax4.hist(data, bins=20, color=colors[2], alpha=0.7, edgecolor='white')
                ax4.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                ax4.axvline(data.median(), color='blue', linestyle='--', linewidth=2, label='Median')
                ax4.legend()
                ax4.set_xlabel(primary_col.title())
                ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def _create_fallback_chart(self, df: pd.DataFrame) -> plt.Figure:
        """Create a fallback chart when other visualizations fail"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            data = df[col].dropna()
            ax.plot(range(len(data)), data.values, color='#667eea', linewidth=2, marker='o', markersize=4)
            ax.set_title(f'📊 {col.title()} Overview', fontsize=16, fontweight='bold')
            ax.set_xlabel('Data Points')
            ax.set_ylabel(col.title())
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '📊 Data Successfully Loaded\nReady for Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16, color='#667eea')
            ax.set_title('Data Overview', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _calculate_executive_kpis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate executive-level KPIs"""
        kpis = []
        numeric_cols = analysis['numeric_columns']
        quality = analysis['data_quality']
        
        # KPI 1: Data Quality Score
        kpis.append({
            'title': 'Data Quality',
            'value': f"{quality['completeness']:.1f}%",
            'subtitle': 'Completeness Score',
            'trend': 'up' if quality['completeness'] > 90 else 'down',
            'trend_value': f"{quality['total_records']:,} records"
        })
        
        # KPI 2: Primary Performance Metric
        if numeric_cols:
            primary_col = numeric_cols[0]
            total_val = df[primary_col].sum()
            avg_val = df[primary_col].mean()
            
            # Smart formatting based on values
            if total_val > 1000000:
                value_display = f"${total_val/1000000:.1f}M"
            elif total_val > 1000:
                value_display = f"${total_val/1000:.1f}K"
            else:
                value_display = f"{total_val:.0f}"
            
            kpis.append({
                'title': primary_col.replace('_', ' ').title(),
                'value': value_display,
                'subtitle': f"Avg: {avg_val:.1f}",
                'trend': 'up',
                'trend_value': f"±{df[primary_col].std():.1f}"
            })
        
        # KPI 3: Insights Generated
        insights_count = len(analysis['correlations']) + len(analysis['patterns']['trends'])
        kpis.append({
            'title': 'AI Insights',
            'value': str(insights_count),
            'subtitle': 'Patterns Found',
            'trend': 'up' if insights_count > 3 else 'neutral',
            'trend_value': 'Auto-detected'
        })
        
        # KPI 4: Business Intelligence Score
        bi_score = min(100, (quality['completeness'] + len(analysis['correlations'])*10 + len(numeric_cols)*5))
        kpis.append({
            'title': 'BI Score',
            'value': f"{bi_score:.0f}",
            'subtitle': 'Intelligence Rating',
            'trend': 'up' if bi_score > 70 else 'down',
            'trend_value': f"{analysis['business_domain'].title()}"
        })
        
        return kpis
    
    def _create_error_dashboard(self, error_message: str) -> Dict[str, Any]:
        """Create an error dashboard when analysis fails"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'⚠️ Dashboard Generation Error\n\n{error_message}\n\nPlease try uploading a different CSV file', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Error Dashboard', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return {
            'charts': [{'title': 'Error', 'chart': fig, 'type': 'error', 'description': 'Error occurred during analysis'}],
            'executive_metrics': [
                {'title': 'Status', 'value': 'Error', 'subtitle': 'Failed', 'trend': 'down', 'trend_value': 'Retry'},
                {'title': 'Data', 'value': 'N/A', 'subtitle': 'Invalid', 'trend': 'down', 'trend_value': 'Check'},
                {'title': 'Charts', 'value': '0', 'subtitle': 'Generated', 'trend': 'down', 'trend_value': 'None'},
                {'title': 'Action', 'value': '⚠️', 'subtitle': 'Required', 'trend': 'down', 'trend_value': 'Upload'}
            ],
            'ai_insights': f'Dashboard generation failed: {error_message}',
            'analysis': {},
            'dashboard_type': 'error'
        }
    
    def _explain_chart_selection(self, chart_type: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Explain why the AI chose this specific chart type"""
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        domain = analysis['business_domain']
        correlations = analysis['correlations']
        patterns = analysis['patterns']
        
        explanations = {
            'executive_summary': f"""
🧠 **AI Chart Selection Reasoning:**
I chose an Executive Summary Dashboard because:
• **Business Context**: Detected {domain} domain data requiring high-level overview
• **Data Composition**: Found {len(numeric_cols)} numeric metrics suitable for KPI display
• **Executive Need**: Multi-panel layout provides comprehensive business intelligence
• **Correlation Analysis**: {len(correlations)} relationships identified for strategic insights
• **Trend Detection**: {len(patterns.get('trends', []))} trends found requiring executive attention
            """.strip(),
            
            'advanced_analytics': f"""
🧠 **AI Chart Selection Reasoning:**
I selected Advanced Analytics visualization because:
• **Statistical Depth**: {len(numeric_cols)} numeric variables enable sophisticated analysis
• **Correlation Strength**: {len(correlations)} significant relationships detected (>0.5 correlation)
• **Pattern Complexity**: Data shows statistical patterns requiring advanced visualization
• **Outlier Detection**: {len(patterns.get('outliers', {}))} variables contain outliers needing analysis
• **Research Value**: Complex relationships benefit from network and regression analysis
            """.strip(),
            
            'performance_intelligence': f"""
🧠 **AI Chart Selection Reasoning:**
I created Performance Intelligence charts because:
• **Category Analysis**: {len(categorical_cols)} categorical variables enable performance comparison
• **Benchmarking Need**: Multiple categories require ranking and performance matrix analysis
• **Business Intelligence**: {domain} domain benefits from category-wise performance insights
• **Volume vs Quality**: Data structure supports performance vs volume bubble analysis
• **Distribution Analysis**: Categories show varying performance distributions requiring violin plots
            """.strip(),
            
            'predictive_insights': f"""
🧠 **AI Chart Selection Reasoning:**
I generated Predictive Insights because:
• **Trend Analysis**: {len(patterns.get('trends', []))} significant trends detected in data
• **Forecasting Value**: Time-series patterns enable statistical forecasting
• **Control Charts**: Data variability requires statistical process control analysis
• **Business Planning**: {domain} domain benefits from predictive intelligence
• **Risk Assessment**: Moving statistics help identify emerging patterns and anomalies
            """.strip()
        }
        
        return explanations.get(chart_type, "Chart selected based on data characteristics and business intelligence requirements.")
    
    def _generate_detailed_chart_explanation(self, chart_type: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate detailed explanation of what each chart shows"""
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        domain = analysis['business_domain']
        
        explanations = {
            'executive_summary': f"""
📊 **Detailed Chart Analysis:**

**Panel 1 - Key Metrics Overview**: Shows average values of {len(numeric_cols)} primary metrics. The bar heights represent performance levels, with taller bars indicating higher values. This helps executives quickly identify top-performing metrics.

**Panel 2 - Trend Analysis**: Displays the trend line for {numeric_cols[0] if numeric_cols else 'primary metric'} over time. The blue line shows actual data points, while the dashed line indicates the overall trend direction (upward/downward).

**Panel 3 - Distribution Analysis**: Histogram showing how {numeric_cols[0] if numeric_cols else 'values'} are distributed. The red dashed line shows the mean (average), while the blue line shows the median (middle value). This reveals data concentration patterns.

**Panel 4 - Category Performance**: Pie chart breaking down performance by {categorical_cols[0] if categorical_cols else 'categories'}. Each slice represents the relative contribution of each category to total performance.

**Panel 5 - Correlation Matrix**: Heatmap showing relationships between all numeric variables. Red indicates negative correlation, blue indicates positive correlation. Stronger colors mean stronger relationships.
            """.strip(),
            
            'advanced_analytics': f"""
📊 **Detailed Chart Analysis:**

**Correlation Network**: Visual network showing how variables connect. Lines between nodes represent correlations >0.3. Thicker lines indicate stronger relationships. Green lines show positive correlations, red lines show negative correlations.

**Distribution Comparison**: Overlapping histograms comparing the distribution shapes of {numeric_cols[0] if len(numeric_cols) > 0 else 'variable 1'} vs {numeric_cols[1] if len(numeric_cols) > 1 else 'variable 2'}. Different colors help identify distribution differences and overlaps.

**Outlier Analysis**: Box plots showing data spread and outliers for each variable. The box shows the middle 50% of data, whiskers show the range, and dots represent outliers that fall outside normal patterns.

**Relationship Analysis**: Scatter plot with regression line showing the relationship between two key variables. The red dashed line shows the trend, and R² value indicates relationship strength (closer to 1 = stronger relationship).
            """.strip(),
            
            'performance_intelligence': f"""
📊 **Detailed Chart Analysis:**

**Performance Ranking**: Horizontal bar chart ranking {categorical_cols[0] if categorical_cols else 'categories'} by average performance. Longer bars indicate better performance. Numbers at the end show exact values.

**Performance vs Volume**: Bubble chart where X-axis shows volume (count), Y-axis shows performance level, and bubble size represents total impact. This identifies high-impact, high-performance categories.

**Performance Distribution**: Violin plots showing the full distribution of performance within each category. Wider sections indicate more common values, helping identify consistency vs variability.

**Performance Matrix**: Scatter plot positioning categories by performance level (X-axis) vs consistency (Y-axis). Four quadrants help identify: High Performance/Low Variability (ideal), High Performance/High Variability (inconsistent winners), etc.
            """.strip(),
            
            'predictive_insights': f"""
📊 **Detailed Chart Analysis:**

**Trend Forecasting**: Shows historical data (solid line) and projected future values (dashed line). The shaded area represents confidence intervals - wider bands indicate less certainty in predictions.

**Statistical Control Chart**: Monitors process stability with control limits (dashed lines). Points outside the control limits (marked with X) indicate unusual events requiring investigation.

**Moving Statistics**: Shows rolling averages and standard deviations to smooth out short-term fluctuations and reveal underlying trends. This helps identify gradual changes that might be missed in raw data.

**Seasonal Patterns**: Identifies recurring patterns in the data that repeat over time periods. This helps predict future behavior based on historical cycles and seasonal effects.
            """.strip()
        }
        
        return explanations.get(chart_type, "This chart provides insights into your data patterns and business metrics.")