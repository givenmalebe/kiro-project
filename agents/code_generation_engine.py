"""
Code Generation Engine for AI-Powered Chart Creation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from typing import Dict, List, Any, Optional
import ast
import sys
import io
import warnings

warnings.filterwarnings('ignore')

class CodeGenerationEngine:
    """Engine for generating and executing safe Python visualization code"""
    
    def __init__(self):
        self.chart_templates = self._load_comprehensive_templates()
        self.safe_globals = self._create_safe_execution_environment()
        self.color_palettes = self._initialize_color_palettes()
    
    def generate_visualization_code(self, chart_spec: Dict[str, Any]) -> str:
        """Generate Python code for a specific chart specification"""
        chart_type = chart_spec.get('chart_type', 'default')
        data_context = chart_spec.get('data_context', {})
        styling = chart_spec.get('styling', {})
        
        # Try to get a creative template first
        if chart_type in ['performance_dashboard', 'correlation_analysis', 'category_performance', 'distribution_analysis']:
            template = self._get_creative_template(chart_type, data_context)
        else:
            template = self.chart_templates.get(chart_type, self.chart_templates['default'])
        
        # Customize template with data context
        customized_code = self._customize_template(template, data_context, styling)
        
        return customized_code
    
    def _get_creative_template(self, chart_type: str, data_context: Dict) -> str:
        """Get creative chart templates for AI-generated visualizations"""
        numeric_cols = data_context.get('numeric_columns', [])
        categorical_cols = data_context.get('categorical_columns', [])
        
        creative_templates = {
            'performance_dashboard': f'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Create a sophisticated performance dashboard
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Set the overall style
try:
    try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

# Main title
fig.suptitle('🚀 AI-Generated Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)

# Chart 1: Key Metrics Overview (Top row, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
if len(df.select_dtypes(include=['number']).columns) > 0:
    numeric_data = df.select_dtypes(include=['number'])
    metrics = numeric_data.mean()
    bars = ax1.bar(range(len(metrics)), metrics.values, color=colors[:len(metrics)])
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels([col.replace('_', '\\n').title() for col in metrics.index], rotation=0)
    ax1.set_title('📊 Key Performance Metrics', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values)*0.01,
                f'{{value:.1f}}', ha='center', va='bottom', fontweight='bold')

# Chart 2: Trend Analysis (Top right)
ax2 = fig.add_subplot(gs[0, 2:])
if len(df.select_dtypes(include=['number']).columns) > 0:
    col = df.select_dtypes(include=['number']).columns[0]
    data = df[col].dropna()
    ax2.plot(range(len(data)), data.values, color=colors[0], linewidth=3, marker='o', markersize=6)
    ax2.fill_between(range(len(data)), data.values, alpha=0.3, color=colors[0])
    ax2.set_title(f'📈 {{col.title()}} Trend', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

# Chart 3: Distribution Analysis (Middle left)
ax3 = fig.add_subplot(gs[1, :2])
if len(df.select_dtypes(include=['number']).columns) > 0:
    col = df.select_dtypes(include=['number']).columns[0]
    data = df[col].dropna()
    n, bins, patches = ax3.hist(data, bins=25, alpha=0.8, color=colors[1], edgecolor='white')
    
    # Color gradient for histogram
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.viridis(i / len(patches)))
    
    ax3.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {{data.mean():.2f}}')
    ax3.set_title(f'📊 {{col.title()}} Distribution', fontsize=14, fontweight='bold')
    ax3.legend()

# Chart 4: Category Analysis (Middle right)
ax4 = fig.add_subplot(gs[1, 2:])
if len(df.select_dtypes(include=['object']).columns) > 0 and len(df.select_dtypes(include=['number']).columns) > 0:
    cat_col = df.select_dtypes(include=['object']).columns[0]
    num_col = df.select_dtypes(include=['number']).columns[0]
    category_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(8)
    
    wedges, texts, autotexts = ax4.pie(category_data.values, labels=category_data.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title(f'🎯 {{cat_col.title()}} Breakdown', fontsize=14, fontweight='bold')

# Chart 5: Correlation Heatmap (Bottom row)
ax5 = fig.add_subplot(gs[2, :])
if len(df.select_dtypes(include=['number']).columns) > 1:
    corr_matrix = df.select_dtypes(include=['number']).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={{"shrink": .8}}, ax=ax5)
    ax5.set_title('🔗 Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
            ''',
            
            'correlation_analysis': f'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🔗 AI-Generated Correlation Analysis', fontsize=18, fontweight='bold')

numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    # Chart 1: Enhanced Correlation Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={{"shrink": .8}}, ax=ax1)
    ax1.set_title('🔥 Correlation Heatmap', fontweight='bold')
    
    # Chart 2: Correlation Network
    G = nx.Graph()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr_val))
    
    if len(G.nodes()) > 0:
        pos = nx.spring_layout(G)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=8, font_weight='bold',
                edge_color=weights, edge_cmap=plt.cm.viridis, width=3)
        ax2.set_title('🕸️ Correlation Network', fontweight='bold')
    
    # Chart 3: Strongest Correlations Bar Chart
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.1:
                corr_pairs.append((f'{{corr_matrix.columns[i]}} vs {{corr_matrix.columns[j]}}', corr_val))
    
    corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    if corr_pairs:
        pairs, values = zip(*corr_pairs[:8])
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax3.barh(range(len(pairs)), values, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(pairs)))
        ax3.set_yticklabels([p[:20] + '...' if len(p) > 20 else p for p in pairs])
        ax3.set_title('📊 Strongest Correlations', fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Chart 4: Scatter Plot of Strongest Correlation
    if corr_pairs:
        strongest_pair = corr_pairs[0][0].split(' vs ')
        if len(strongest_pair) == 2:
            col1, col2 = strongest_pair[0], strongest_pair[1]
            if col1 in df.columns and col2 in df.columns:
                ax4.scatter(df[col1], df[col2], alpha=0.6, color='#667eea', s=50)
                ax4.set_xlabel(col1)
                ax4.set_ylabel(col2)
                ax4.set_title(f'💫 {{col1}} vs {{col2}}', fontweight='bold')
                
                # Add trend line
                z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
                p = np.poly1d(z)
                ax4.plot(df[col1], p(df[col1]), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
            ''',
            
            'category_performance': f'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🎯 AI-Generated Category Performance Analysis', fontsize=18, fontweight='bold')

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=['number']).columns

if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    
    # Group data
    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count', 'std']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False).head(10)
    
    # Chart 1: Performance Bar Chart with Gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
    bars = ax1.bar(range(len(grouped)), grouped['mean'], color=colors)
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels(grouped[cat_col], rotation=45, ha='right')
    ax1.set_title('📊 Average Performance by Category', fontweight='bold')
    ax1.set_ylabel(f'Average {{num_col.title()}}')
    
    # Add value labels
    for bar, value in zip(bars, grouped['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grouped['mean'])*0.01,
                f'{{value:.1f}}', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Bubble Chart (Size = Count, Color = Performance)
    scatter = ax2.scatter(grouped['sum'], grouped['mean'], s=grouped['count']*20, 
                         c=grouped['mean'], cmap='viridis', alpha=0.7, edgecolors='white', linewidth=2)
    ax2.set_xlabel(f'Total {{num_col.title()}}')
    ax2.set_ylabel(f'Average {{num_col.title()}}')
    ax2.set_title('🎈 Performance Bubble Chart', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Performance Level')
    
    # Add category labels
    for i, row in grouped.iterrows():
        ax2.annotate(row[cat_col][:8], (row['sum'], row['mean']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    
    # Chart 3: Box Plot Distribution
    categories = grouped[cat_col].head(6)
    box_data = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
    bp = ax3.boxplot(box_data, labels=[cat[:8] for cat in categories], patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(categories)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_title('📦 Distribution by Category', fontweight='bold')
    ax3.set_ylabel(f'{{num_col.title()}}')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Chart 4: Performance Ranking
    top_performers = grouped.head(5)
    y_pos = np.arange(len(top_performers))
    bars = ax4.barh(y_pos, top_performers['mean'], color='gold', alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_performers[cat_col])
    ax4.set_title('🏆 Top Performers', fontweight='bold')
    ax4.set_xlabel(f'Average {{num_col.title()}}')
    
    # Add ranking numbers
    for i, (bar, value) in enumerate(zip(bars, top_performers['mean'])):
        ax4.text(bar.get_width() + max(top_performers['mean'])*0.01, bar.get_y() + bar.get_height()/2,
                f'#{{i+1}} ({{value:.1f}})', va='center', fontweight='bold')

plt.tight_layout()
            ''',
            
            'distribution_analysis': f'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 AI-Generated Distribution Analysis', fontsize=18, fontweight='bold')

numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    
    # Chart 1: Enhanced Histogram with Statistics
    n, bins, patches = ax1.hist(data, bins=30, alpha=0.7, edgecolor='white', linewidth=1.2)
    
    # Color bars based on quartiles
    q25, q50, q75 = data.quantile([0.25, 0.5, 0.75])
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val < q25:
            patch.set_facecolor('#ff6b6b')  # Red for bottom quartile
        elif bin_val < q50:
            patch.set_facecolor('#feca57')  # Yellow for second quartile
        elif bin_val < q75:
            patch.set_facecolor('#48dbfb')  # Blue for third quartile
        else:
            patch.set_facecolor('#0be881')  # Green for top quartile
    
    # Add statistical lines
    mean_val = data.mean()
    median_val = data.median()
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=3, label=f'Mean: {{mean_val:.2f}}')
    ax1.axvline(median_val, color='blue', linestyle='--', linewidth=3, label=f'Median: {{median_val:.2f}}')
    ax1.set_title(f'📊 {{col.title()}} Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Q-Q Plot for Normality
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('📈 Q-Q Plot (Normality Test)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Box Plot with Outlier Analysis
    bp = ax3.boxplot(data, patch_artist=True, notch=True, showfliers=True)
    bp['boxes'][0].set_facecolor('#667eea')
    bp['boxes'][0].set_alpha(0.7)
    
    # Identify outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
    
    ax3.set_title(f'📦 Box Plot Analysis\\n{{len(outliers)}} outliers detected', fontweight='bold')
    ax3.set_ylabel(f'{{col.title()}}')
    
    # Chart 4: Density Plot with Multiple Distributions
    ax4.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='Data')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (μ={{mu:.2f}}, σ={{sigma:.2f}})')
    
    # Add KDE
    kde = stats.gaussian_kde(data)
    ax4.plot(x, kde(x), 'g-', linewidth=2, label='KDE')
    
    ax4.set_title('📈 Density Analysis', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
            '''
        }
        
        return creative_templates.get(chart_type, self.chart_templates['default'])
    
    def validate_code_safety(self, code: str) -> bool:
        """Validate that code is safe for execution"""
        try:
            tree = ast.parse(code)
            
            # Whitelist of allowed modules and functions
            allowed_modules = {
                'matplotlib', 'matplotlib.pyplot', 'numpy', 'pandas', 
                'seaborn', 'plotly', 'plotly.graph_objects', 'plotly.express'
            }
            
            allowed_functions = {
                'len', 'max', 'min', 'sum', 'abs', 'round', 'int', 'float',
                'str', 'list', 'dict', 'tuple', 'range', 'enumerate', 'zip',
                'sorted', 'reversed', 'print'
            }
            
            # Check all nodes in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in allowed_modules:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in allowed_modules:
                        return False
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in allowed_functions and not node.func.id.startswith(('plt.', 'ax.', 'fig.', 'df.', 'np.', 'pd.', 'sns.')):
                            return False
            
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def execute_code_safely(self, code: str, data: pd.DataFrame) -> Optional[matplotlib.figure.Figure]:
        """Execute visualization code in a safe environment"""
        if not self.validate_code_safety(code):
            return None
        
        try:
            # Create safe execution environment
            safe_locals = {
                'df': data,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'go': go,
                'px': px,
                'fig': None,
                'ax': None
            }
            
            # Capture output to prevent unwanted prints
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Execute the code
                exec(code, self.safe_globals, safe_locals)
                return safe_locals.get('fig')
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            print(f"Code execution error: {e}")
            return None
    
    def get_chart_template(self, chart_type: str) -> str:
        """Get a specific chart template"""
        return self.chart_templates.get(chart_type, self.chart_templates['default'])
    
    def _customize_template(self, template: str, data_context: Dict, styling: Dict) -> str:
        """Customize a template with specific data context and styling"""
        customized = template
        
        # Replace column placeholders
        if 'numeric_columns' in data_context and data_context['numeric_columns']:
            customized = customized.replace('NUMERIC_COL_1', data_context['numeric_columns'][0])
            if len(data_context['numeric_columns']) > 1:
                customized = customized.replace('NUMERIC_COL_2', data_context['numeric_columns'][1])
        
        if 'categorical_columns' in data_context and data_context['categorical_columns']:
            customized = customized.replace('CATEGORICAL_COL_1', data_context['categorical_columns'][0])
        
        # Apply styling
        business_domain = data_context.get('business_domain', 'general')
        colors = self.color_palettes.get(business_domain, self.color_palettes['general'])
        customized = customized.replace('PRIMARY_COLOR', colors['primary'])
        customized = customized.replace('SECONDARY_COLOR', colors['secondary'])
        
        return customized
    
    def _create_safe_execution_environment(self) -> Dict[str, Any]:
        """Create a restricted globals environment for safe code execution"""
        import builtins
        safe_builtins = {}
        
        # Allow safe built-in functions
        safe_functions = [
            'len', 'max', 'min', 'sum', 'abs', 'round', 'int', 'float',
            'str', 'list', 'dict', 'tuple', 'range', 'enumerate', 'zip',
            'sorted', 'reversed', 'print', 'type', 'isinstance', 'hasattr',
            'getattr', 'setattr', 'bool', 'any', 'all'
        ]
        
        for func_name in safe_functions:
            if hasattr(builtins, func_name):
                safe_builtins[func_name] = getattr(builtins, func_name)
        
        return {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'go': go,
            'px': px
        }
    
    def _initialize_color_palettes(self) -> Dict[str, Dict[str, str]]:
        """Initialize color palettes for different business domains"""
        return {
            'finance': {
                'primary': '#2E8B57',
                'secondary': '#FFD700',
                'accent': '#FF6347',
                'palette': ['#2E8B57', '#FFD700', '#FF6347', '#4169E1', '#32CD32']
            },
            'marketing': {
                'primary': '#FF6B6B',
                'secondary': '#4ECDC4',
                'accent': '#45B7D1',
                'palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            },
            'sales': {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'accent': '#f093fb',
                'palette': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            },
            'operations': {
                'primary': '#3498db',
                'secondary': '#e74c3c',
                'accent': '#2ecc71',
                'palette': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            },
            'general': {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'accent': '#f093fb',
                'palette': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            }
        }
    
    def _load_comprehensive_templates(self) -> Dict[str, str]:
        """Load comprehensive chart templates for different visualization types"""
        return {
            'financial_performance': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Financial Performance Dashboard', fontsize=16, fontweight='bold')

# Revenue Trend (Top Left)
if 'NUMERIC_COL_1' in df.columns:
    data = df['NUMERIC_COL_1'].dropna()
    ax1.plot(range(len(data)), data.values, color='PRIMARY_COLOR', linewidth=3, marker='o')
    ax1.set_title('Revenue Trend', fontweight='bold')
    ax1.set_ylabel('Revenue ($)')
    ax1.grid(True, alpha=0.3)
    
    # Add moving average
    if len(data) > 5:
        window = min(7, len(data) // 3)
        ma = data.rolling(window=window).mean()
        ax1.plot(range(len(ma)), ma.values, '--', color='SECONDARY_COLOR', linewidth=2, label=f'{window}-MA')
        ax1.legend()

# Distribution Analysis (Top Right)
if 'NUMERIC_COL_1' in df.columns:
    ax2.hist(df['NUMERIC_COL_1'].dropna(), bins=20, alpha=0.7, color='PRIMARY_COLOR', edgecolor='white')
    ax2.set_title('Revenue Distribution', fontweight='bold')
    ax2.set_xlabel('Revenue ($)')
    ax2.set_ylabel('Frequency')
    
    # Add statistics
    mean_val = df['NUMERIC_COL_1'].mean()
    ax2.axvline(mean_val, color='SECONDARY_COLOR', linestyle='--', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
    ax2.legend()

# Category Performance (Bottom Left)
if 'CATEGORICAL_COL_1' in df.columns and 'NUMERIC_COL_1' in df.columns:
    cat_performance = df.groupby('CATEGORICAL_COL_1')['NUMERIC_COL_1'].mean().sort_values(ascending=False).head(8)
    bars = ax3.bar(range(len(cat_performance)), cat_performance.values, color='PRIMARY_COLOR', alpha=0.8)
    ax3.set_title('Performance by Category', fontweight='bold')
    ax3.set_xticks(range(len(cat_performance)))
    ax3.set_xticklabels(cat_performance.index, rotation=45, ha='right')
    ax3.set_ylabel('Average Performance')
    
    # Add value labels
    for bar, value in zip(bars, cat_performance.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cat_performance.values)*0.01,
                f'${value:,.0f}', ha='center', va='bottom', fontsize=9)

# KPI Summary (Bottom Right)
if len(df.select_dtypes(include=['number']).columns) > 0:
    numeric_cols = df.select_dtypes(include=['number']).columns
    kpis = []
    for col in numeric_cols[:4]:
        kpis.append(f"{col}: ${df[col].sum():,.0f}")
    
    ax4.text(0.1, 0.8, "Key Performance Indicators", fontsize=14, fontweight='bold', transform=ax4.transAxes)
    for i, kpi in enumerate(kpis):
        ax4.text(0.1, 0.6 - i*0.15, f"• {kpi}", fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')

plt.tight_layout()
            ''',
            
            'correlation_heatmap': '''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))

numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Highlight strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corrs.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    if strong_corrs:
        ax.text(0.02, 0.98, "Strong Correlations (>0.7):", transform=ax.transAxes, 
                fontsize=10, fontweight='bold', va='top')
        for i, corr in enumerate(strong_corrs[:3]):
            ax.text(0.02, 0.92 - i*0.06, f"• {corr}", transform=ax.transAxes, 
                    fontsize=9, va='top')
else:
    ax.text(0.5, 0.5, 'Correlation Analysis\\nRequires Multiple Numeric Columns', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Correlation Analysis')

plt.tight_layout()
            ''',
            
            'trend_analysis': '''
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Advanced Trend Analysis', fontsize=16, fontweight='bold')

numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    
    # Main trend plot (Top)
    x = range(len(data))
    ax1.plot(x, data.values, color='PRIMARY_COLOR', linewidth=2, alpha=0.8, label='Actual Data')
    
    # Add trend line with confidence interval
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
    trend_line = slope * np.array(x) + intercept
    ax1.plot(x, trend_line, '--', color='SECONDARY_COLOR', linewidth=3, label=f'Trend (R²={r_value**2:.3f})')
    
    # Confidence interval
    residuals = data.values - trend_line
    mse = np.mean(residuals**2)
    confidence = 1.96 * np.sqrt(mse)
    ax1.fill_between(x, trend_line - confidence, trend_line + confidence, 
                     alpha=0.2, color='SECONDARY_COLOR', label='95% Confidence')
    
    ax1.set_title(f'{col} - Trend Analysis with Statistical Confidence', fontweight='bold')
    ax1.set_ylabel(col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Moving averages (Bottom)
    if len(data) > 10:
        ma_short = data.rolling(window=5).mean()
        ma_long = data.rolling(window=min(20, len(data)//2)).mean()
        
        ax2.plot(x, data.values, color='lightgray', alpha=0.5, label='Original')
        ax2.plot(x, ma_short.values, color='PRIMARY_COLOR', linewidth=2, label='5-Period MA')
        ax2.plot(x, ma_long.values, color='SECONDARY_COLOR', linewidth=2, label=f'{min(20, len(data)//2)}-Period MA')
        
        ax2.set_title('Moving Averages Comparison', fontweight='bold')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel(col)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for moving averages\\n(Need >10 data points)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Moving Averages Analysis')

plt.tight_layout()
            ''',
            
            'business_kpi_dashboard': '''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Business KPI Dashboard', fontsize=18, fontweight='bold', y=0.95)

# KPI Cards (Top Row)
numeric_cols = df.select_dtypes(include=['number']).columns
kpi_colors = ['PRIMARY_COLOR', 'SECONDARY_COLOR', '#f093fb', '#f5576c']

for i, col in enumerate(numeric_cols[:4]):
    ax = fig.add_subplot(gs[0, i])
    
    # Calculate KPI value
    total_val = df[col].sum()
    avg_val = df[col].mean()
    
    # Create KPI card
    ax.text(0.5, 0.7, f'${total_val:,.0f}', ha='center', va='center', 
            transform=ax.transAxes, fontsize=20, fontweight='bold', color=kpi_colors[i])
    ax.text(0.5, 0.4, col.replace('_', ' ').title(), ha='center', va='center',
            transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, 0.2, f'Avg: ${avg_val:,.0f}', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, alpha=0.8)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor(kpi_colors[i])
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

# Performance Chart (Middle Left)
ax_perf = fig.add_subplot(gs[1, :2])
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    ax_perf.plot(range(len(data)), data.values, color='PRIMARY_COLOR', linewidth=3, marker='o', markersize=4)
    ax_perf.set_title('Performance Trend', fontweight='bold', fontsize=14)
    ax_perf.set_ylabel(col)
    ax_perf.grid(True, alpha=0.3)

# Category Analysis (Middle Right)
ax_cat = fig.add_subplot(gs[1, 2:])
if 'CATEGORICAL_COL_1' in df.columns and len(numeric_cols) > 0:
    cat_data = df.groupby('CATEGORICAL_COL_1')[numeric_cols[0]].mean().sort_values(ascending=False).head(6)
    bars = ax_cat.bar(range(len(cat_data)), cat_data.values, color='SECONDARY_COLOR', alpha=0.8)
    ax_cat.set_title('Category Performance', fontweight='bold', fontsize=14)
    ax_cat.set_xticks(range(len(cat_data)))
    ax_cat.set_xticklabels(cat_data.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, cat_data.values):
        ax_cat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cat_data.values)*0.01,
                   f'${value:,.0f}', ha='center', va='bottom', fontsize=9)

# Distribution Analysis (Bottom)
ax_dist = fig.add_subplot(gs[2, :])
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    
    # Create histogram with statistics
    n, bins, patches = ax_dist.hist(data, bins=25, alpha=0.7, color='PRIMARY_COLOR', edgecolor='white')
    
    # Add statistical lines
    mean_val = data.mean()
    median_val = data.median()
    q25, q75 = data.quantile([0.25, 0.75])
    
    ax_dist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
    ax_dist.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: ${median_val:,.0f}')
    ax_dist.axvspan(q25, q75, alpha=0.2, color='green', label='IQR')
    
    ax_dist.set_title('Value Distribution with Key Statistics', fontweight='bold', fontsize=14)
    ax_dist.set_xlabel(col)
    ax_dist.set_ylabel('Frequency')
    ax_dist.legend()

plt.tight_layout()
            ''',
            
            'category_performance': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))

if 'CATEGORICAL_COL_1' in df.columns and 'NUMERIC_COL_1' in df.columns:
    # Group by category and calculate performance metrics
    category_data = df.groupby('CATEGORICAL_COL_1')['NUMERIC_COL_1'].agg(['mean', 'sum', 'count']).reset_index()
    category_data = category_data.sort_values('mean', ascending=False).head(10)
    
    # Create horizontal bar chart for better readability
    y_pos = np.arange(len(category_data))
    bars = ax.barh(y_pos, category_data['mean'], color='PRIMARY_COLOR', alpha=0.8, height=0.6)
    
    # Customize the chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(category_data['CATEGORICAL_COL_1'])
    ax.set_xlabel('Average Performance')
    ax.set_title('Performance by Category', fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value, count) in enumerate(zip(bars, category_data['mean'], category_data['count'])):
        ax.text(bar.get_width() + max(category_data['mean'])*0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.1f} (n={count})', va='center', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Color bars based on performance (gradient)
    max_val = category_data['mean'].max()
    for bar, value in zip(bars, category_data['mean']):
        intensity = value / max_val
        bar.set_color(plt.cm.viridis(intensity))
else:
    ax.text(0.5, 0.5, 'Category Performance Analysis\\nReady for Data', 
            ha='center', va='center', transform=ax.transAxes, fontsize=16)
    ax.set_title('Category Performance')

plt.tight_layout()
            ''',
            
            'default': '''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))

# Create a professional-looking placeholder
ax.text(0.5, 0.6, '📊 AI Dashboard', ha='center', va='center', 
        transform=ax.transAxes, fontsize=24, fontweight='bold', color='PRIMARY_COLOR')
ax.text(0.5, 0.4, 'Analyzing your data...\\nGenerating insights...', ha='center', va='center',
        transform=ax.transAxes, fontsize=14, color='gray')

# Add some visual elements
circle = plt.Circle((0.5, 0.5), 0.3, fill=False, color='PRIMARY_COLOR', linewidth=3, transform=ax.transAxes)
ax.add_patch(circle)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Dashboard Ready', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
            '''
        }