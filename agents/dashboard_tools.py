"""
Advanced AI-Powered Dashboard Tools for Intelligent Layout and Visualization Creation
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import warnings
import colorsys
import math

warnings.filterwarnings('ignore')

class DashboardTools:
    """Advanced AI-Powered tools for creating intelligent dashboards with stunning layouts"""
    
    # Color palettes for different business contexts
    COLOR_PALETTES = {
        'executive': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
        'financial': ['#2E8B57', '#FFD700', '#FF6347', '#4169E1', '#32CD32', '#FF4500'],
        'marketing': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
        'operations': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
        'healthcare': ['#27ae60', '#3498db', '#e67e22', '#e74c3c', '#9b59b6', '#f1c40f'],
        'education': ['#8e44ad', '#3498db', '#e67e22', '#27ae60', '#e74c3c', '#f39c12']
    }
    
    @staticmethod
    def create_ai_dashboard_layout(df: pd.DataFrame, business_domain: str = 'general') -> Dict[str, Any]:
        """AI-powered dashboard layout creation with intelligent component placement"""
        
        # Analyze data structure for optimal layout
        layout_analysis = DashboardTools._analyze_data_for_layout(df)
        
        # Generate intelligent layout configuration
        layout_config = {
            'grid_system': DashboardTools._determine_optimal_grid(layout_analysis),
            'component_hierarchy': DashboardTools._create_component_hierarchy(layout_analysis),
            'color_scheme': DashboardTools._generate_intelligent_colors(business_domain, layout_analysis),
            'typography': DashboardTools._select_typography_system(business_domain),
            'spacing': DashboardTools._calculate_optimal_spacing(layout_analysis),
            'responsive_breakpoints': DashboardTools._define_responsive_breakpoints(),
            'animation_config': DashboardTools._create_animation_config(layout_analysis)
        }
        
        return layout_config
    
    @staticmethod
    def _analyze_data_for_layout(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform layout decisions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        analysis = {
            'data_complexity': len(df.columns),
            'record_count': len(df),
            'numeric_ratio': len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0,
            'categorical_ratio': len(categorical_cols) / len(df.columns) if len(df.columns) > 0 else 0,
            'has_time_series': len(datetime_cols) > 0,
            'data_density': df.size,
            'missing_data_ratio': df.isnull().sum().sum() / df.size if df.size > 0 else 0,
            'correlation_strength': DashboardTools._calculate_correlation_strength(df, numeric_cols),
            'outlier_presence': DashboardTools._detect_outlier_presence(df, numeric_cols),
            'distribution_types': DashboardTools._analyze_distributions(df, numeric_cols)
        }
        
        return analysis
    
    @staticmethod
    def _calculate_correlation_strength(df: pd.DataFrame, numeric_cols: List[str]) -> float:
        """Calculate overall correlation strength in the dataset"""
        if len(numeric_cols) < 2:
            return 0.0
        
        try:
            corr_matrix = df[numeric_cols].corr()
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            correlations = upper_triangle.stack().abs()
            return correlations.mean() if not correlations.empty else 0.0
        except:
            return 0.0
    
    @staticmethod
    def _detect_outlier_presence(df: pd.DataFrame, numeric_cols: List[str]) -> float:
        """Detect presence of outliers using IQR method"""
        if len(numeric_cols) == 0:
            return 0.0
        
        outlier_ratios = []
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_ratio = len(outliers) / len(df) if len(df) > 0 else 0
                outlier_ratios.append(outlier_ratio)
            except:
                outlier_ratios.append(0.0)
        
        return np.mean(outlier_ratios) if outlier_ratios else 0.0
    
    @staticmethod
    def _analyze_distributions(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, str]:
        """Analyze distribution types for numeric columns"""
        distributions = {}
        
        for col in numeric_cols:
            try:
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()
                
                if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                    distributions[col] = 'normal'
                elif skewness > 1:
                    distributions[col] = 'right_skewed'
                elif skewness < -1:
                    distributions[col] = 'left_skewed'
                elif kurtosis > 3:
                    distributions[col] = 'heavy_tailed'
                else:
                    distributions[col] = 'irregular'
            except:
                distributions[col] = 'unknown'
        
        return distributions
    
    @staticmethod
    def _determine_optimal_grid(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal grid system based on data analysis"""
        complexity = analysis['data_complexity']
        record_count = analysis['record_count']
        
        if complexity <= 5 and record_count <= 1000:
            grid_type = 'simple_2x2'
            columns = 2
            rows = 2
        elif complexity <= 10 and record_count <= 10000:
            grid_type = 'standard_3x2'
            columns = 3
            rows = 2
        elif complexity <= 20:
            grid_type = 'complex_4x2'
            columns = 4
            rows = 2
        else:
            grid_type = 'enterprise_4x3'
            columns = 4
            rows = 3
        
        return {
            'type': grid_type,
            'columns': columns,
            'rows': rows,
            'gap': '1rem',
            'responsive': True,
            'auto_fit': True
        }
    
    @staticmethod
    def _create_component_hierarchy(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create intelligent component hierarchy based on data characteristics"""
        components = []
        
        # KPI Cards (Always top priority)
        components.append({
            'type': 'kpi_cards',
            'priority': 1,
            'position': 'top',
            'span': {'columns': 4, 'rows': 1},
            'count': 4,
            'style': 'executive'
        })
        
        # Primary Chart (Based on data characteristics)
        if analysis['has_time_series']:
            primary_chart = 'time_series_advanced'
        elif analysis['correlation_strength'] > 0.5:
            primary_chart = 'correlation_matrix'
        elif analysis['categorical_ratio'] > 0.3:
            primary_chart = 'market_analysis'
        else:
            primary_chart = 'performance_trend'
        
        components.append({
            'type': 'primary_chart',
            'chart_type': primary_chart,
            'priority': 2,
            'position': 'main',
            'span': {'columns': 2, 'rows': 1},
            'interactive': True,
            'animations': True
        })
        
        # Secondary Charts
        components.extend([
            {
                'type': 'distribution_chart',
                'priority': 3,
                'position': 'secondary',
                'span': {'columns': 2, 'rows': 1},
                'chart_type': 'enhanced_histogram'
            },
            {
                'type': 'comparison_chart',
                'priority': 4,
                'position': 'secondary',
                'span': {'columns': 2, 'rows': 1},
                'chart_type': 'competitive_analysis'
            },
            {
                'type': 'insight_chart',
                'priority': 5,
                'position': 'secondary',
                'span': {'columns': 2, 'rows': 1},
                'chart_type': 'opportunity_matrix'
            }
        ])
        
        return components
    
    @staticmethod
    def _generate_intelligent_colors(business_domain: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent color schemes based on business domain and data characteristics"""
        base_palette = DashboardTools.COLOR_PALETTES.get(business_domain, DashboardTools.COLOR_PALETTES['executive'])
        
        # Adjust colors based on data characteristics
        if analysis['correlation_strength'] > 0.7:
            # High correlation - use gradient colors
            color_scheme = DashboardTools._create_gradient_palette(base_palette[0], base_palette[1], 6)
        elif analysis['outlier_presence'] > 0.1:
            # High outliers - use contrasting colors
            color_scheme = DashboardTools._create_contrasting_palette(base_palette)
        else:
            # Standard harmonious palette
            color_scheme = base_palette
        
        return {
            'primary': color_scheme[0],
            'secondary': color_scheme[1],
            'accent': color_scheme[2],
            'palette': color_scheme,
            'gradients': DashboardTools._create_gradient_definitions(color_scheme),
            'semantic': {
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'info': '#17a2b8'
            }
        }
    
    @staticmethod
    def _create_gradient_palette(color1: str, color2: str, steps: int) -> List[str]:
        """Create gradient color palette between two colors"""
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        
        gradient = []
        for i in range(steps):
            ratio = i / (steps - 1) if steps > 1 else 0
            r = rgb1[0] + (rgb2[0] - rgb1[0]) * ratio
            g = rgb1[1] + (rgb2[1] - rgb1[1]) * ratio
            b = rgb1[2] + (rgb2[2] - rgb1[2]) * ratio
            gradient.append(rgb_to_hex((r, g, b)))
        
        return gradient
    
    @staticmethod
    def _create_contrasting_palette(base_palette: List[str]) -> List[str]:
        """Create high-contrast color palette for outlier visualization"""
        contrasting = []
        for color in base_palette:
            # Convert to HSV and adjust saturation/value for contrast
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            hsv = colorsys.rgb_to_hsv(*rgb)
            
            # Increase saturation and adjust value for contrast
            new_hsv = (hsv[0], min(1.0, hsv[1] * 1.3), max(0.3, min(0.9, hsv[2] * 1.1)))
            new_rgb = colorsys.hsv_to_rgb(*new_hsv)
            new_hex = '#{:02x}{:02x}{:02x}'.format(
                int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255)
            )
            contrasting.append(new_hex)
        
        return contrasting
    
    @staticmethod
    def _create_gradient_definitions(color_scheme: List[str]) -> Dict[str, str]:
        """Create CSS gradient definitions"""
        gradients = {}
        
        for i, color in enumerate(color_scheme[:3]):  # Use first 3 colors
            next_color = color_scheme[(i + 1) % len(color_scheme)]
            gradients[f'gradient_{i+1}'] = f'linear-gradient(135deg, {color} 0%, {next_color} 100%)'
        
        # Special gradients
        gradients['primary_gradient'] = f'linear-gradient(135deg, {color_scheme[0]} 0%, {color_scheme[1]} 100%)'
        gradients['success_gradient'] = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)'
        gradients['warning_gradient'] = 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)'
        
        return gradients
    
    @staticmethod
    def _select_typography_system(business_domain: str) -> Dict[str, Any]:
        """Select appropriate typography system for business domain"""
        typography_systems = {
            'executive': {
                'primary_font': 'Inter, system-ui, sans-serif',
                'secondary_font': 'Georgia, serif',
                'heading_weight': 700,
                'body_weight': 400,
                'line_height': 1.6,
                'scale': 1.25  # Major third scale
            },
            'financial': {
                'primary_font': 'Roboto, Arial, sans-serif',
                'secondary_font': 'Roboto Mono, monospace',
                'heading_weight': 600,
                'body_weight': 400,
                'line_height': 1.5,
                'scale': 1.2  # Minor third scale
            },
            'marketing': {
                'primary_font': 'Poppins, sans-serif',
                'secondary_font': 'Open Sans, sans-serif',
                'heading_weight': 600,
                'body_weight': 400,
                'line_height': 1.7,
                'scale': 1.33  # Perfect fourth scale
            }
        }
        
        return typography_systems.get(business_domain, typography_systems['executive'])
    
    @staticmethod
    def _calculate_optimal_spacing(analysis: Dict[str, Any]) -> Dict[str, str]:
        """Calculate optimal spacing based on data complexity"""
        complexity = analysis['data_complexity']
        
        if complexity <= 5:
            spacing = {
                'xs': '0.25rem',
                'sm': '0.5rem',
                'md': '1rem',
                'lg': '1.5rem',
                'xl': '2rem'
            }
        elif complexity <= 15:
            spacing = {
                'xs': '0.375rem',
                'sm': '0.75rem',
                'md': '1.25rem',
                'lg': '2rem',
                'xl': '2.5rem'
            }
        else:
            spacing = {
                'xs': '0.5rem',
                'sm': '1rem',
                'md': '1.5rem',
                'lg': '2.5rem',
                'xl': '3rem'
            }
        
        return spacing
    
    @staticmethod
    def _define_responsive_breakpoints() -> Dict[str, str]:
        """Define responsive breakpoints for dashboard layout"""
        return {
            'mobile': '480px',
            'tablet': '768px',
            'desktop': '1024px',
            'large': '1200px',
            'xlarge': '1440px'
        }
    
    @staticmethod
    def _create_animation_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create animation configuration based on data characteristics"""
        record_count = analysis['record_count']
        
        # Adjust animation complexity based on data size
        if record_count > 10000:
            animation_duration = 300  # Faster for large datasets
            easing = 'ease-out'
        elif record_count > 1000:
            animation_duration = 500  # Medium speed
            easing = 'ease-in-out'
        else:
            animation_duration = 800  # Slower, more detailed animations
            easing = 'cubic-bezier(0.4, 0, 0.2, 1)'
        
        return {
            'duration': animation_duration,
            'easing': easing,
            'stagger': 100,  # Delay between animated elements
            'entrance': 'fadeInUp',
            'hover': 'scale',
            'loading': 'pulse'
        }
    
    @staticmethod
    def detect_business_domain(df: pd.DataFrame) -> Dict[str, Any]:
        """AI-powered business domain detection"""
        domain_indicators = {
            'ecommerce': ['order', 'product', 'customer', 'price', 'quantity', 'cart', 'purchase'],
            'finance': ['revenue', 'profit', 'cost', 'investment', 'return', 'portfolio', 'balance'],
            'marketing': ['campaign', 'conversion', 'click', 'impression', 'lead', 'acquisition'],
            'sales': ['deal', 'opportunity', 'pipeline', 'quota', 'commission', 'territory'],
            'hr': ['employee', 'salary', 'department', 'performance', 'training', 'retention'],
            'operations': ['production', 'inventory', 'supply', 'logistics', 'efficiency', 'quality'],
            'healthcare': ['patient', 'treatment', 'diagnosis', 'medication', 'hospital', 'doctor'],
            'education': ['student', 'course', 'grade', 'enrollment', 'teacher', 'curriculum']
        }
        
        column_names = [col.lower() for col in df.columns]
        domain_scores = {}
        
        for domain, keywords in domain_indicators.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in column_names))
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain] / len(domain_indicators[primary_domain])
            return {
                'primary_domain': primary_domain,
                'confidence': confidence,
                'all_scores': domain_scores
            }
        
        return {'primary_domain': 'general', 'confidence': 0.0, 'all_scores': {}}
    
    @staticmethod
    def create_executive_summary(df: pd.DataFrame, metrics: Dict, domain: str) -> str:
        """Generate executive summary based on business domain"""
        summary_templates = {
            'ecommerce': """
            **E-commerce Performance Summary**
            
            📊 **Business Overview**: {records:,} transactions analyzed across {columns} data points
            🎯 **Key Metrics**: Average order value, customer segments, product performance
            💰 **Revenue Insights**: {revenue_insight}
            📈 **Growth Opportunities**: {growth_opportunities}
            ⚠️ **Risk Factors**: {risk_factors}
            """,
            
            'finance': """
            **Financial Performance Summary**
            
            💼 **Portfolio Overview**: {records:,} financial records across {columns} metrics
            📊 **Key Indicators**: Revenue trends, profitability, cost analysis
            💰 **Financial Health**: {revenue_insight}
            📈 **Investment Opportunities**: {growth_opportunities}
            ⚠️ **Risk Assessment**: {risk_factors}
            """,
            
            'general': """
            **Business Intelligence Summary**
            
            📊 **Data Overview**: {records:,} records analyzed across {columns} dimensions
            🎯 **Key Insights**: {key_insights}
            💰 **Performance Metrics**: {revenue_insight}
            📈 **Opportunities**: {growth_opportunities}
            ⚠️ **Areas of Concern**: {risk_factors}
            """
        }
        
        template = summary_templates.get(domain, summary_templates['general'])
        
        # Extract insights from metrics
        revenue_cols = [k for k in metrics.keys() if any(word in k.lower() for word in ['revenue', 'sales', 'amount'])]
        revenue_insight = f"Primary revenue metric shows strong performance" if revenue_cols else "Revenue metrics not clearly identified"
        
        growth_opportunities = "Data shows potential for optimization in key areas"
        risk_factors = "Monitor data quality and variability in key metrics"
        key_insights = "Strong data foundation with actionable business metrics"
        
        return template.format(
            records=metrics.get('total_records', 0),
            columns=metrics.get('total_columns', 0),
            revenue_insight=revenue_insight,
            growth_opportunities=growth_opportunities,
            risk_factors=risk_factors,
            key_insights=key_insights
        )
    
    @staticmethod
    def create_smart_kpis(df: pd.DataFrame, domain: str) -> List[Dict]:
        """Create domain-specific KPIs"""
        kpi_templates = {
            'ecommerce': [
                {'name': 'Average Order Value', 'calc': 'mean', 'format': 'currency'},
                {'name': 'Conversion Rate', 'calc': 'percentage', 'format': 'percentage'},
                {'name': 'Customer Lifetime Value', 'calc': 'sum', 'format': 'currency'},
                {'name': 'Cart Abandonment Rate', 'calc': 'percentage', 'format': 'percentage'}
            ],
            'finance': [
                {'name': 'Total Revenue', 'calc': 'sum', 'format': 'currency'},
                {'name': 'Profit Margin', 'calc': 'percentage', 'format': 'percentage'},
                {'name': 'ROI', 'calc': 'percentage', 'format': 'percentage'},
                {'name': 'Cost Efficiency', 'calc': 'ratio', 'format': 'number'}
            ],
            'general': [
                {'name': 'Data Quality Score', 'calc': 'quality', 'format': 'percentage'},
                {'name': 'Performance Index', 'calc': 'mean', 'format': 'number'},
                {'name': 'Growth Rate', 'calc': 'growth', 'format': 'percentage'},
                {'name': 'Efficiency Ratio', 'calc': 'ratio', 'format': 'number'}
            ]
        }
        
        templates = kpi_templates.get(domain, kpi_templates['general'])
        kpis = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for template in templates:
            # Find best matching column for this KPI
            matching_col = None
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in template['name'].lower().split()):
                    matching_col = col
                    break
            
            if matching_col:
                if template['calc'] == 'mean':
                    value = df[matching_col].mean()
                elif template['calc'] == 'sum':
                    value = df[matching_col].sum()
                elif template['calc'] == 'percentage':
                    value = (df[matching_col].mean() / df[matching_col].max()) * 100 if df[matching_col].max() != 0 else 0
                elif template['calc'] == 'quality':
                    value = ((df.size - df.isnull().sum().sum()) / df.size) * 100
                elif template['calc'] == 'growth':
                    if len(df) > 1:
                        value = ((df[matching_col].iloc[-1] - df[matching_col].iloc[0]) / df[matching_col].iloc[0]) * 100 if df[matching_col].iloc[0] != 0 else 0
                    else:
                        value = 0
                else:
                    value = df[matching_col].mean()
                
                kpis.append({
                    'name': template['name'],
                    'value': value,
                    'format': template['format'],
                    'column': matching_col
                })
        
        return kpis
    
    @staticmethod
    def create_stunning_chart(df: pd.DataFrame, chart_type: str, config: Dict) -> Optional[go.Figure]:
        """Create stunning, AI-optimized charts with advanced visual design"""
        try:
            chart_creators = {
                'financial_performance_advanced': DashboardTools._create_financial_performance_chart,
                'market_intelligence_advanced': DashboardTools._create_market_intelligence_chart,
                'profitability_matrix_advanced': DashboardTools._create_profitability_matrix_chart,
                'growth_opportunities_advanced': DashboardTools._create_growth_opportunities_chart,
                'executive_summary_advanced': DashboardTools._create_executive_summary_chart,
                'competitive_landscape': DashboardTools._create_competitive_landscape_chart,
                'performance_trends': DashboardTools._create_performance_trends_chart,
                'risk_assessment_matrix': DashboardTools._create_risk_assessment_chart,
                'opportunity_heatmap': DashboardTools._create_opportunity_heatmap_chart,
                'customer_segmentation': DashboardTools._create_customer_segmentation_chart
            }
            
            creator_func = chart_creators.get(chart_type, DashboardTools._create_default_advanced_chart)
            return creator_func(df, config)
            
        except Exception as e:
            print(f"Error creating stunning chart {chart_type}: {e}")
            return DashboardTools._create_fallback_chart(df, config)
    
    @staticmethod
    def _create_financial_performance_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create advanced financial performance visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return DashboardTools._create_fallback_chart(df, config)
        
        # Find financial columns
        financial_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                         for keyword in ['revenue', 'sales', 'profit', 'income', 'amount', 'price'])]
        primary_col = financial_cols[0] if financial_cols else numeric_cols[0]
        
        # Create multi-dimensional financial analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Trend & Forecast',
                'Distribution Analysis',
                'Benchmark Comparison',
                'Growth Trajectory'
            ),
            specs=[
                [{"secondary_y": True}, {"type": "histogram"}],
                [{"type": "indicator"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Performance trend with advanced analytics
        df_sorted = df.sort_index()
        
        # Main performance line with gradient fill
        fig.add_trace(go.Scatter(
            x=df_sorted.index,
            y=df_sorted[primary_col],
            mode='lines+markers',
            name='Performance',
            line=dict(color='#667eea', width=4, shape='spline'),
            marker=dict(size=8, color='#667eea', line=dict(width=2, color='white')),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>Performance</b><br>Period: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        
        # Moving average with confidence bands
        if len(df_sorted) > 5:
            window = min(7, len(df_sorted) // 3)
            ma = df_sorted[primary_col].rolling(window=window).mean()
            ma_std = df_sorted[primary_col].rolling(window=window).std()
            
            fig.add_trace(go.Scatter(
                x=df_sorted.index,
                y=ma,
                mode='lines',
                name='Trend',
                line=dict(color='#f093fb', width=3, dash='dash'),
            ), row=1, col=1)
            
            # Confidence bands
            upper_band = ma + (ma_std * 1.96)
            lower_band = ma - (ma_std * 1.96)
            
            fig.add_trace(go.Scatter(
                x=df_sorted.index, y=upper_band, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_sorted.index, y=lower_band, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(240, 147, 251, 0.1)',
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
        
        # 2. Enhanced distribution with statistical overlays
        fig.add_trace(go.Histogram(
            x=df[primary_col],
            nbinsx=25,
            name='Distribution',
            marker=dict(
                color='rgba(102, 126, 234, 0.7)',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Distribution</b><br>Range: $%{x}<br>Count: %{y}<extra></extra>'
        ), row=1, col=2)
        
        # Add statistical markers
        mean_val = df[primary_col].mean()
        median_val = df[primary_col].median()
        
        fig.add_vline(x=mean_val, line_dash="solid", line_color="#f5576c", line_width=2,
                     annotation_text=f"Mean: ${mean_val:,.0f}", row=1, col=2)
        fig.add_vline(x=median_val, line_dash="dash", line_color="#4facfe", line_width=2,
                     annotation_text=f"Median: ${median_val:,.0f}", row=1, col=2)
        
        # 3. Performance gauge
        current_performance = df[primary_col].mean()
        max_performance = df[primary_col].max()
        performance_ratio = (current_performance / max_performance) * 100 if max_performance > 0 else 0
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=performance_ratio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Performance Score"},
            delta={'reference': 80, 'increasing': {'color': "#28a745"}, 'decreasing': {'color': "#dc3545"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 80], 'color': "#fff3e0"},
                    {'range': [80, 100], 'color': "#e8f5e8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=1)
        
        # 4. Growth trajectory analysis
        if len(df) > 1:
            # Calculate period-over-period growth
            growth_rates = df[primary_col].pct_change().fillna(0) * 100
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=growth_rates,
                mode='lines+markers',
                name='Growth Rate',
                line=dict(color='#43e97b', width=3),
                marker=dict(size=6, color=growth_rates, colorscale='RdYlGn', 
                           showscale=True, colorbar=dict(title="Growth %", x=1.02)),
                hovertemplate='<b>Growth Rate</b><br>Period: %{x}<br>Growth: %{y:.1f}%<extra></extra>'
            ), row=2, col=2)
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
        
        # Enhanced styling
        fig.update_layout(
            title=dict(
                text=f"💰 {primary_col.title()} - Advanced Financial Performance Analytics",
                font=dict(size=18, color='#2c3e50', family='Inter, sans-serif'),
                x=0.5
            ),
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            font=dict(family="Inter, sans-serif", size=11, color='#2c3e50'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1
            )
        )
        
        return fig
    
    @staticmethod
    def _create_market_intelligence_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create advanced market intelligence visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return DashboardTools._create_fallback_chart(df, config)
        
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Advanced market analysis
        market_data = df.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count', 'std']).reset_index()
        market_data = market_data.sort_values('sum', ascending=False).head(10)
        
        # Calculate market metrics
        total_market = market_data['sum'].sum()
        market_data['market_share'] = (market_data['sum'] / total_market * 100)
        market_data['efficiency'] = market_data['sum'] / market_data['count']
        market_data['volatility'] = (market_data['std'] / market_data['mean'] * 100).fillna(0)
        
        # Create comprehensive market visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Market Share & Concentration',
                'Performance vs Volume Matrix',
                'Competitive Positioning',
                'Market Efficiency Analysis'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15
        )
        
        # 1. Enhanced donut chart with market concentration
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', 
                 '#43e97b', '#38f9d7', '#ffecd2', '#fcb69f']
        
        fig.add_trace(go.Pie(
            labels=market_data[cat_col],
            values=market_data['sum'],
            hole=0.5,
            marker=dict(colors=colors[:len(market_data)], line=dict(color='white', width=2)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        ), row=1, col=1)
        
        # 2. Performance efficiency matrix
        fig.add_trace(go.Scatter(
            x=market_data['count'],
            y=market_data['efficiency'],
            mode='markers+text',
            text=market_data[cat_col],
            textposition='top center',
            marker=dict(
                size=market_data['market_share'] * 2 + 10,
                color=market_data['volatility'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Volatility %", x=0.48),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>Volume: %{x}<br>Efficiency: $%{y:,.0f}<br>Volatility: %{marker.color:.1f}%<extra></extra>'
        ), row=1, col=2)
        
        # 3. Market share bars with gradient
        fig.add_trace(go.Bar(
            x=market_data[cat_col],
            y=market_data['market_share'],
            marker=dict(
                color=market_data['market_share'],
                colorscale='viridis',
                line=dict(color='white', width=1)
            ),
            text=[f"{share:.1f}%" for share in market_data['market_share']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Market Share: %{y:.1f}%<extra></extra>'
        ), row=2, col=1)
        
        # 4. Efficiency vs market share analysis
        fig.add_trace(go.Scatter(
            x=market_data['market_share'],
            y=market_data['efficiency'],
            mode='markers+text',
            text=market_data[cat_col],
            textposition='top center',
            marker=dict(
                size=market_data['sum'] / market_data['sum'].max() * 30 + 10,
                color=market_data['count'],
                colorscale='plasma',
                showscale=True,
                colorbar=dict(title="Volume", x=1.02),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Market Share: %{x:.1f}%<br>Efficiency: $%{y:,.0f}<extra></extra>'
        ), row=2, col=2)
        
        # Enhanced styling
        fig.update_layout(
            title=dict(
                text=f"🎯 {cat_col.title()} - Market Intelligence & Competitive Analysis",
                font=dict(size=18, color='#2c3e50', family='Inter, sans-serif'),
                x=0.5
            ),
            height=600,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            font=dict(family="Inter, sans-serif", size=11, color='#2c3e50'),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _create_fallback_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a fallback chart when specific chart creation fails"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines+markers',
                name=col.title(),
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea')
            ))
            
            fig.update_layout(
                title=f"📊 {col.title()} Analysis",
                height=400,
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)'
            )
            
            return fig
        else:
            # Create a simple info chart
            fig = go.Figure()
            fig.add_annotation(
                text="Data visualization ready<br>Upload numeric data for advanced charts",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color='#6c757d')
            )
            fig.update_layout(
                title="📊 Dashboard Ready",
                height=400,
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)'
            )
            return fig
    
    @staticmethod
    def _create_default_advanced_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create default advanced chart with AI-optimized styling"""
        return DashboardTools._create_fallback_chart(df, config)
    
    @staticmethod
    def _create_executive_dashboard(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a comprehensive executive dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trends', 'Performance Metrics', 'Risk Analysis', 'Key Indicators'),
            specs=[[{"secondary_y": True}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
        
        if len(numeric_cols) >= 1:
            # Revenue trend
            fig.add_trace(
                go.Scatter(x=df.index, y=df[numeric_cols[0]], name=numeric_cols[0]),
                row=1, col=1
            )
        
        if len(numeric_cols) >= 2:
            # Performance indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=df[numeric_cols[1]].mean(),
                    title={'text': numeric_cols[1]},
                    gauge={'axis': {'range': [df[numeric_cols[1]].min(), df[numeric_cols[1]].max()]}}
                ),
                row=1, col=2
            )
        
        if len(numeric_cols) >= 3:
            # Risk scatter
            fig.add_trace(
                go.Scatter(
                    x=df[numeric_cols[2]], 
                    y=df[numeric_cols[0]] if len(numeric_cols) >= 1 else df[numeric_cols[2]],
                    mode='markers',
                    name='Risk Analysis'
                ),
                row=2, col=1
            )
        
        if len(numeric_cols) >= 4:
            # Key indicators bar
            fig.add_trace(
                go.Bar(x=df.index[:10], y=df[numeric_cols[3]][:10], name=numeric_cols[3]),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Executive Dashboard Overview")
        return fig
    
    @staticmethod
    def _create_performance_matrix(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a performance matrix heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Performance Correlation Matrix",
            height=500
        )
        
        return fig
    
    @staticmethod
    def _create_trend_forecast(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create trend analysis with simple forecasting"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        target_col = numeric_cols[0]
        
        # Simple moving average forecast
        window = min(7, len(df) // 4)
        if window < 2:
            window = 2
        
        df_copy = df.copy()
        df_copy['MA'] = df_copy[target_col].rolling(window=window).mean()
        
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[target_col],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df_copy['MA'],
            mode='lines',
            name=f'{window}-Period MA',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Trend Analysis: {target_col}",
            xaxis_title="Index",
            yaxis_title=target_col,
            height=400
        )
        
        return fig
    
    @staticmethod
    def _create_risk_heatmap(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create risk assessment heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        # Calculate risk metrics (coefficient of variation)
        risk_data = []
        for col in numeric_cols:
            if df[col].std() > 0 and df[col].mean() != 0:
                cv = (df[col].std() / df[col].mean()) * 100
                risk_level = 'High' if cv > 50 else 'Medium' if cv > 25 else 'Low'
                risk_data.append({
                    'Metric': col,
                    'Risk_Score': cv,
                    'Risk_Level': risk_level
                })
        
        if not risk_data:
            return None
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = go.Figure(data=go.Bar(
            x=risk_df['Metric'],
            y=risk_df['Risk_Score'],
            marker_color=['red' if x > 50 else 'orange' if x > 25 else 'green' for x in risk_df['Risk_Score']],
            text=risk_df['Risk_Level'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Risk Assessment by Metric",
            xaxis_title="Metrics",
            yaxis_title="Risk Score (CV %)",
            height=400
        )
        
        return fig
    
    @staticmethod
    def _create_correlation_network(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create correlation network visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 3:
            return None
        
        # Calculate correlations
        corr_matrix = df[numeric_cols].corr()
        
        # Create network data
        edges = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.iloc[i, j]) > 0.3:  # Only strong correlations
                    edges.append({
                        'source': col1,
                        'target': col2,
                        'weight': abs(corr_matrix.iloc[i, j])
                    })
        
        if not edges:
            return None
        
        # Simple network layout (circular)
        import math
        n_nodes = len(numeric_cols)
        node_positions = {}
        for i, col in enumerate(numeric_cols):
            angle = 2 * math.pi * i / n_nodes
            node_positions[col] = (math.cos(angle), math.sin(angle))
        
        # Create traces for edges
        edge_trace = []
        for edge in edges:
            x0, y0 = node_positions[edge['source']]
            x1, y1 = node_positions[edge['target']]
            edge_trace.extend([x0, x1, None])
            edge_trace.extend([y0, y1, None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_trace[::3],
            y=edge_trace[1::3],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))
        
        # Add nodes
        node_x = [node_positions[col][0] for col in numeric_cols]
        node_y = [node_positions[col][1] for col in numeric_cols]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=numeric_cols,
            textposition='middle center',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Correlation Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    
    @staticmethod
    def export_dashboard_config(dashboard_data: Dict, filename: str = None) -> Dict:
        """Export dashboard configuration for reuse"""
        if filename is None:
            filename = f"dashboard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        config = {
            'timestamp': datetime.now().isoformat(),
            'metrics': dashboard_data.get('metrics', {}),
            'chart_configs': [],
            'business_context': dashboard_data.get('business_context', ''),
            'ai_insights': dashboard_data.get('ai_insights', '')
        }
        
        # Extract chart configurations
        for chart in dashboard_data.get('charts', []):
            config['chart_configs'].append({
                'type': chart.get('type'),
                'title': chart.get('title'),
                'story': chart.get('story')
            })
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            return {'success': True, 'filename': filename, 'config': config}
        except Exception as e:
            return {'success': False, 'error': str(e), 'config': config}
    
    @staticmethod
    def _create_profitability_matrix_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create advanced profitability matrix visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return DashboardTools._create_fallback_chart(df, config)
        
        # Use first two numeric columns for profitability analysis
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        
        # Calculate medians for quadrant division
        x_median = df[x_col].median()
        y_median = df[y_col].median()
        
        # Create profitability matrix
        fig = go.Figure()
        
        # Color code by quadrant performance
        colors = []
        quadrant_labels = []
        for _, row in df.iterrows():
            if row[x_col] >= x_median and row[y_col] >= y_median:
                colors.append('#2E8B57')  # Green - High performance
                quadrant_labels.append('Star Performer')
            elif row[x_col] >= x_median and row[y_col] < y_median:
                colors.append('#FFD700')  # Gold - High efficiency, low revenue
                quadrant_labels.append('Efficient')
            elif row[x_col] < x_median and row[y_col] >= y_median:
                colors.append('#FF6347')  # Red - Low efficiency, high revenue
                quadrant_labels.append('High Volume')
            else:
                colors.append('#808080')  # Gray - Needs improvement
                quadrant_labels.append('Needs Focus')
        
        # Scatter plot with quadrant analysis
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                color=colors,
                size=10,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=quadrant_labels,
            hovertemplate=f'<b>%{{text}}</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>',
            name='Performance Quadrants'
        ))
        
        # Add quadrant lines
        fig.add_vline(x=x_median, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=y_median, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(x=df[x_col].max()*0.8, y=df[y_col].max()*0.9, 
                         text="⭐ Stars", showarrow=False, font=dict(color="green", size=12))
        fig.add_annotation(x=df[x_col].max()*0.8, y=df[y_col].min()*1.1, 
                         text="⚡ Efficient", showarrow=False, font=dict(color="orange", size=12))
        fig.add_annotation(x=df[x_col].min()*1.1, y=df[y_col].max()*0.9, 
                         text="📈 Volume", showarrow=False, font=dict(color="red", size=12))
        fig.add_annotation(x=df[x_col].min()*1.1, y=df[y_col].min()*1.1, 
                         text="🎯 Focus", showarrow=False, font=dict(color="gray", size=12))
        
        fig.update_layout(
            title=f"💎 Profitability Matrix: {x_col.title()} vs {y_col.title()}",
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title(),
            height=400,
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _create_growth_opportunities_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create growth opportunities analysis chart"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return DashboardTools._create_fallback_chart(df, config)
        
        primary_metric = numeric_cols[0]
        
        # Create opportunity analysis with actionable insights
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Performance Distribution', 'Growth Opportunities'),
            specs=[[{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Left: Performance distribution with targets
        fig.add_trace(go.Histogram(
            x=df[primary_metric],
            nbinsx=20,
            name='Current Performance',
            marker_color='#4ECDC4',
            opacity=0.7
        ), row=1, col=1)
        
        # Add performance benchmarks
        q25 = df[primary_metric].quantile(0.25)
        q75 = df[primary_metric].quantile(0.75)
        q90 = df[primary_metric].quantile(0.90)
        
        fig.add_vline(x=q75, line_dash="dash", line_color="orange", 
                     annotation_text="Top 25%", row=1, col=1)
        fig.add_vline(x=q90, line_dash="dash", line_color="red", 
                     annotation_text="Top 10%", row=1, col=1)
        
        # Right: Opportunity analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            opp_data = df.groupby(cat_col)[primary_metric].agg(['mean', 'max', 'count']).reset_index()
            opp_data['opportunity_score'] = (opp_data['max'] - opp_data['mean']) * opp_data['count']
            opp_data = opp_data.sort_values('opportunity_score', ascending=True).tail(8)
            
            fig.add_trace(go.Bar(
                y=opp_data[cat_col],
                x=opp_data['opportunity_score'],
                orientation='h',
                name='Opportunity Score',
                marker_color='#FF6B6B',
                text=[f"${val:,.0f}" for val in opp_data['opportunity_score']],
                textposition='outside'
            ), row=1, col=2)
        else:
            # Fallback: Show top vs bottom performance
            top_performers = df[df[primary_metric] >= q75]
            bottom_performers = df[df[primary_metric] <= q25]
            
            performance_data = pd.DataFrame({
                'Category': ['Bottom 25%', 'Middle 50%', 'Top 25%'],
                'Count': [len(bottom_performers), len(df) - len(top_performers) - len(bottom_performers), len(top_performers)],
                'Avg_Value': [bottom_performers[primary_metric].mean() if len(bottom_performers) > 0 else 0,
                             df[(df[primary_metric] > q25) & (df[primary_metric] < q75)][primary_metric].mean(),
                             top_performers[primary_metric].mean() if len(top_performers) > 0 else 0]
            })
            
            fig.add_trace(go.Bar(
                x=performance_data['Category'],
                y=performance_data['Avg_Value'],
                name='Average Performance',
                marker_color=['#FF6B6B', '#FFD93D', '#6BCF7F'],
                text=[f"${val:,.0f}" for val in performance_data['Avg_Value']],
                textposition='outside'
            ), row=1, col=2)
        
        fig.update_layout(
            title="🚀 Strategic Growth Opportunities & Performance Analysis",
            height=400,
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            showlegend=False
        )
        
        fig.update_xaxes(title_text=primary_metric.title(), row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Opportunity Score", row=1, col=2)
        
        return fig
    
    @staticmethod
    def _create_executive_summary_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create executive summary chart"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return DashboardTools._create_fallback_chart(df, config)
        
        # Create summary metrics chart
        summary_data = df[numeric_cols[:4]].describe().T if len(numeric_cols) >= 4 else df[numeric_cols].describe().T
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=summary_data.index,
            y=summary_data['mean'],
            name='Mean',
            marker_color='#667eea',
            text=[f"{val:,.0f}" for val in summary_data['mean']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="📊 Executive Summary - Key Metrics Overview",
            height=400,
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            xaxis_title="Metrics",
            yaxis_title="Average Values"
        )
        
        return fig
    
    @staticmethod
    def _create_competitive_landscape_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create competitive landscape visualization"""
        return DashboardTools._create_market_intelligence_chart(df, config)
    
    @staticmethod
    def _create_performance_trends_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create performance trends visualization"""
        return DashboardTools._create_financial_performance_chart(df, config)
    
    @staticmethod
    def _create_risk_assessment_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create risk assessment matrix"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return DashboardTools._create_fallback_chart(df, config)
        
        # Calculate risk metrics (coefficient of variation)
        risk_data = []
        for col in numeric_cols:
            if df[col].std() > 0 and df[col].mean() != 0:
                cv = (df[col].std() / df[col].mean()) * 100
                risk_level = 'High' if cv > 50 else 'Medium' if cv > 25 else 'Low'
                risk_data.append({
                    'Metric': col,
                    'Risk_Score': cv,
                    'Risk_Level': risk_level
                })
        
        if not risk_data:
            return DashboardTools._create_fallback_chart(df, config)
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = go.Figure(data=go.Bar(
            x=risk_df['Metric'],
            y=risk_df['Risk_Score'],
            marker_color=['red' if x > 50 else 'orange' if x > 25 else 'green' for x in risk_df['Risk_Score']],
            text=risk_df['Risk_Level'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="⚠️ Risk Assessment by Metric",
            xaxis_title="Metrics",
            yaxis_title="Risk Score (CV %)",
            height=400,
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)'
        )
        
        return fig
    
    @staticmethod
    def _create_opportunity_heatmap_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create opportunity heatmap visualization"""
        return DashboardTools._create_growth_opportunities_chart(df, config)
    
    @staticmethod
    def _create_customer_segmentation_chart(df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create customer segmentation visualization"""
        return DashboardTools._create_market_intelligence_chart(df, config)