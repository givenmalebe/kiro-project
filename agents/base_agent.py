import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from typing import Any, Dict, Optional, List, Union
import os
import ast
import sys
import io
import contextlib
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

class BaseAgent:
    """Enhanced base class for all AI agents with code generation capabilities"""
    
    def __init__(self, model_name: str = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Use environment variable for model name, fallback to parameter or default
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        
        # Initialize code generation capabilities
        self.chart_templates = self._initialize_chart_templates()
        self.safe_globals = self._create_safe_globals()
        self.execution_timeout = 30  # seconds
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    **kwargs
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced dataframe analysis with pattern detection"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Calculate correlations for numeric columns
        correlations = {}
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Get strong correlations (>0.5)
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        correlations[f"{corr_matrix.columns[i]}_vs_{corr_matrix.columns[j]}"] = corr_val
        
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "correlations": correlations,
            "data_quality_score": ((df.size - df.isnull().sum().sum()) / df.size * 100) if df.size > 0 else 0,
            "outlier_columns": self._detect_outlier_columns(df, numeric_cols),
            "business_domain": self._detect_business_domain(df)
        }
    
    def _detect_outlier_columns(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Detect columns with significant outliers"""
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        return outlier_cols
    
    def _detect_business_domain(self, df: pd.DataFrame) -> str:
        """Detect business domain from column names"""
        domain_keywords = {
            'ecommerce': ['order', 'product', 'customer', 'price', 'quantity', 'cart', 'purchase'],
            'finance': ['revenue', 'profit', 'cost', 'investment', 'return', 'balance', 'amount'],
            'marketing': ['campaign', 'conversion', 'click', 'impression', 'lead', 'acquisition'],
            'sales': ['deal', 'opportunity', 'pipeline', 'quota', 'commission', 'territory'],
            'hr': ['employee', 'salary', 'department', 'performance', 'training', 'retention'],
            'operations': ['production', 'inventory', 'supply', 'logistics', 'efficiency', 'quality']
        }
        
        column_names = [col.lower() for col in df.columns]
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in column_names))
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def generate_chart_code(self, chart_type: str, data_context: Dict[str, Any]) -> str:
        """Generate Python code for creating charts"""
        try:
            # Get base template
            template = self.chart_templates.get(chart_type, self.chart_templates['default'])
            
            # Use AI to customize the template based on data context
            customization_prompt = f"""
            Customize this Python chart code template for the given data context:
            
            Template: {template}
            
            Data Context:
            - Columns: {data_context.get('columns', [])}
            - Numeric columns: {data_context.get('numeric_columns', [])}
            - Categorical columns: {data_context.get('categorical_columns', [])}
            - Chart type: {chart_type}
            - Business domain: {data_context.get('business_domain', 'general')}
            
            Requirements:
            1. Replace placeholder column names with actual column names from the data
            2. Use appropriate colors for the business domain
            3. Add proper titles and labels
            4. Include error handling for missing data
            5. Make the chart professional and visually appealing
            6. Return only the Python code, no explanations
            
            The code should create a matplotlib figure and return it as 'fig'.
            """
            
            customized_code = self.generate_response(customization_prompt)
            
            # Clean the response to extract only Python code
            code_lines = []
            in_code_block = False
            for line in customized_code.split('\n'):
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip().startswith('```'):
                    in_code_block = False
                    continue
                elif in_code_block or (not line.strip().startswith('#') and 'import' in line or 'fig' in line or 'plt' in line or 'ax' in line):
                    code_lines.append(line)
            
            final_code = '\n'.join(code_lines) if code_lines else template
            
            # Validate the generated code
            if self.validate_code_safety(final_code):
                return final_code
            else:
                return template  # Fallback to template if validation fails
                
        except Exception as e:
            print(f"Error generating chart code: {e}")
            return self.chart_templates.get('default', '')
    
    def validate_code_safety(self, code: str) -> bool:
        """Validate that generated code is safe to execute"""
        try:
            # Parse the code to check for dangerous operations
            tree = ast.parse(code)
            
            # Check for dangerous functions/modules
            dangerous_items = [
                'exec', 'eval', 'compile', '__import__', 'open', 'file',
                'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
                'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
            ]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in dangerous_items:
                    return False
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in ['matplotlib', 'numpy', 'pandas', 'seaborn', 'plotly']:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in ['matplotlib.pyplot', 'numpy', 'pandas', 'seaborn', 'plotly.graph_objects', 'plotly.express']:
                        return False
            
            return True
        except:
            return False
    
    def execute_chart_code(self, code: str, df: pd.DataFrame) -> Optional[matplotlib.figure.Figure]:
        """Safely execute chart generation code"""
        try:
            # Create a safe execution environment
            safe_locals = {
                'df': df,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'go': go,
                'px': px,
                'fig': None
            }
            
            # Capture stdout to prevent unwanted output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Execute the code with timeout protection
                exec(code, self.safe_globals, safe_locals)
                return safe_locals.get('fig')
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            print(f"Error executing chart code: {e}")
            return self._create_fallback_chart(df)
    
    def _create_fallback_chart(self, df: pd.DataFrame) -> matplotlib.figure.Figure:
        """Create a simple fallback chart when code execution fails"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Simple histogram of first numeric column
            ax.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7, color='#667eea')
            ax.set_title(f'Distribution of {numeric_cols[0]}')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'Data visualization\nready for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Data Overview')
        
        plt.tight_layout()
        return fig
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe globals dictionary for code execution"""
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
    
    def _initialize_chart_templates(self) -> Dict[str, str]:
        """Initialize chart code templates"""
        return {
            'bar_chart': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
if len(df.select_dtypes(include=['object']).columns) > 0 and len(df.select_dtypes(include=['number']).columns) > 0:
    cat_col = df.select_dtypes(include=['object']).columns[0]
    num_col = df.select_dtypes(include=['number']).columns[0]
    data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
    bars = ax.bar(range(len(data)), data.values, color='#667eea', alpha=0.8)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.set_title(f'{num_col} by {cat_col}')
    ax.set_ylabel(num_col)
    
    # Add value labels on bars
    for bar, value in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data.values)*0.01,
                f'{value:.1f}', ha='center', va='bottom')
else:
    ax.text(0.5, 0.5, 'Bar Chart Ready', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Bar Chart')

plt.tight_layout()
            ''',
            
            'line_chart': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    ax.plot(range(len(data)), data.values, color='#667eea', linewidth=2, marker='o', markersize=4)
    ax.set_title(f'{col} Trend Analysis')
    ax.set_xlabel('Data Points')
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(data)), data.values, 1)
    p = np.poly1d(z)
    ax.plot(range(len(data)), p(range(len(data))), "--", color='#f5576c', alpha=0.8, linewidth=2)
else:
    ax.text(0.5, 0.5, 'Line Chart Ready', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Trend Analysis')

plt.tight_layout()
            ''',
            
            'scatter_plot': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) >= 2:
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    scatter = ax.scatter(df[x_col], df[y_col], alpha=0.6, c='#667eea', s=50)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    
    # Add correlation coefficient
    corr = df[x_col].corr(df[y_col])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add trend line
    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "--", color='#f5576c', alpha=0.8, linewidth=2)
else:
    ax.text(0.5, 0.5, 'Scatter Plot Ready', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Correlation Analysis')

plt.tight_layout()
            ''',
            
            'histogram': '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    col = numeric_cols[0]
    data = df[col].dropna()
    n, bins, patches = ax.hist(data, bins=25, alpha=0.7, color='#667eea', edgecolor='white')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='#f5576c', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#4facfe', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'Histogram Ready', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Distribution Analysis')

plt.tight_layout()
            ''',
            
            'default': '''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.text(0.5, 0.5, 'Chart Ready for Data\\nAnalysis in Progress...', 
        ha='center', va='center', transform=ax.transAxes, fontsize=16, color='#667eea')
ax.set_title('Data Visualization')
ax.axis('off')
plt.tight_layout()
            '''
        }