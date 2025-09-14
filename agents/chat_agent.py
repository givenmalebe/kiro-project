import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from typing import Dict, Any, List

class ChatAgent(BaseAgent):
    """Agent specialized in conversational data analysis - text only, no visualizations"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Data Chat Agent"
        self.conversation_history = []
    
    def chat_with_data(self, df: pd.DataFrame, user_question: str) -> str:
        """Chat with the data - answer user questions about the dataset"""
        
        # Analyze the dataframe for context
        analysis = self.analyze_dataframe(df)
        
        # Get comprehensive statistics and insights
        stats_summary = self._get_data_summary(df)
        detailed_analysis = self._get_detailed_analysis(df)
        
        # Perform specific analysis based on question type
        specific_insights = self.analyze_specific_question(df, user_question)
        
        # Create enhanced context for the AI
        context_prompt = f"""
        You are an expert Data Analysis Assistant with deep knowledge of statistics, business intelligence, and data science. 
        Provide comprehensive, insightful answers to user questions about datasets.
        
        Dataset Information:
        - Shape: {analysis['shape']} (rows, columns)
        - Columns: {analysis['columns']}
        - Numeric columns: {analysis['numeric_columns']}
        - Categorical columns: {analysis['categorical_columns']}
        - Missing values: {analysis['missing_values']}
        
        Comprehensive Data Summary:
        {stats_summary}
        
        Detailed Analysis:
        {detailed_analysis}
        
        Specific Insights for this question:
        {specific_insights}
        
        Sample data (first 5 rows):
        {df.head().to_string()}
        
        User Question: {user_question}
        
        Instructions for your response:
        1. Provide a comprehensive, detailed answer based on the actual data
        2. Include specific statistics, numbers, and data-driven insights
        3. Explain the business implications and practical significance
        4. If applicable, suggest follow-up questions or analyses
        5. Use clear, professional language that's accessible to business users
        6. Reference actual data values and patterns you observe
        7. If the question involves trends, correlations, or patterns, explain them thoroughly
        8. Include actionable insights and recommendations when relevant
        9. If the user asks for visualizations, charts, or graphs, provide Python code that generates the visualization
        10. DO NOT include any HTML tags or technical formatting in the main response - use plain text only
        11. Python code for visualizations should be provided in separate code blocks when requested
        
        Previous conversation context:
        {self._get_conversation_context()}
        """
        
        response = self.generate_response(context_prompt)
        
        # Store conversation history
        self.conversation_history.append({
            "question": user_question,
            "response": response
        })
        
        # Keep only last 5 conversations to manage context length
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        # Check if user is asking for a visualization and execute Python code
        viz_code = self.generate_visualization_code(df, user_question)
        print(f"Generated visualization code: {len(viz_code) if viz_code else 0} characters")
        if viz_code:
            # Execute the code and capture the plot
            plot_image = self._execute_visualization_code(viz_code, df)
            print(f"Generated plot image: {len(plot_image) if plot_image else 0} characters")
            if plot_image:
                response += f"\n\n**📊 Generated Visualization:**"
                # Store the plot image for display in the UI
                response += f"\n\n[PLOT_IMAGE:{plot_image}]"
            else:
                response += f"\n\n**⚠️ Visualization generation failed**"
        
        return response
    
    def _get_data_summary(self, df: pd.DataFrame) -> str:
        """Get a comprehensive data summary"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"\nNumeric columns statistics:")
            for col in numeric_cols:
                stats = df[col].describe()
                summary_parts.append(f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary_parts.append(f"\nCategorical columns info:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                summary_parts.append(f"- {col}: {unique_count} unique values, most common: {most_common}")
        
        return "\n".join(summary_parts)
    
    def _get_detailed_analysis(self, df: pd.DataFrame) -> str:
        """Get detailed analysis including patterns, trends, and insights"""
        analysis_parts = []
        
        # Data quality analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            analysis_parts.append(f"Data Quality: {missing_data.sum()} missing values across {len(missing_data[missing_data > 0])} columns")
        else:
            analysis_parts.append("Data Quality: No missing values found")
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis_parts.append(f"\nNumeric Analysis:")
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    skewness = data.skew()
                    kurtosis = data.kurtosis()
                    analysis_parts.append(f"- {col}: skewness={skewness:.2f}, kurtosis={kurtosis:.2f}")
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append(f"\nCategorical Analysis:")
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                analysis_parts.append(f"- {col}: {len(value_counts)} unique categories, most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.2f}")
            
            if strong_corrs:
                analysis_parts.append(f"\nStrong Correlations (>0.7): {', '.join(strong_corrs[:3])}")
            else:
                analysis_parts.append(f"\nNo strong correlations (>0.7) found between numeric variables")
        
        return "\n".join(analysis_parts)
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "No previous conversation"
        
        context_parts = []
        for conv in self.conversation_history[-3:]:  # Last 3 conversations
            context_parts.append(f"Q: {conv['question']}")
            context_parts.append(f"A: {conv['response'][:200]}...")
        
        return "\n".join(context_parts)
    
    def get_suggested_questions(self, df: pd.DataFrame) -> List[str]:
        """Generate suggested questions based on the dataset"""
        analysis = self.analyze_dataframe(df)
        questions = []
        
        # Basic questions
        questions.append("What are the main characteristics of this dataset?")
        questions.append("What patterns or trends can you identify?")
        questions.append("Are there any outliers or unusual values?")
        
        # Numeric-specific questions
        if len(analysis['numeric_columns']) > 0:
            questions.append(f"What are the statistical properties of the numeric columns?")
            if len(analysis['numeric_columns']) >= 2:
                questions.append("What correlations exist between numeric variables?")
        
        # Categorical-specific questions
        if len(analysis['categorical_columns']) > 0:
            questions.append("What are the most common categories in each categorical column?")
            questions.append("How is the data distributed across different categories?")
        
        # Business-focused questions
        questions.append("What business insights can you derive from this data?")
        questions.append("What recommendations would you make based on this analysis?")
        
        return questions[:8]  # Return up to 8 questions
    
    def analyze_specific_question(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze specific aspects based on the question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['correlation', 'relationship', 'correlate']):
            return self._analyze_correlations(df)
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'unusual']):
            return self._analyze_outliers(df)
        elif any(word in question_lower for word in ['distribution', 'spread', 'pattern']):
            return self._analyze_distributions(df)
        elif any(word in question_lower for word in ['missing', 'null', 'empty']):
            return self._analyze_missing_data(df)
        else:
            return self._general_analysis(df)
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations for correlation-related questions"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numeric columns for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append({
                        'pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                        'correlation': corr_val,
                        'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                    })
        
        return {
            'correlation_count': len(strong_corrs),
            'strong_correlations': strong_corrs[:5],  # Top 5
            'message': f"Found {len(strong_corrs)} significant correlations"
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers for outlier-related questions"""
        numeric_df = df.select_dtypes(include=['number'])
        outliers_info = {}
        
        for col in numeric_df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'values': outliers.tolist()[:5]  # First 5 outlier values
                }
        
        return {
            'outliers_by_column': outliers_info,
            'total_outliers': sum(info['count'] for info in outliers_info.values()),
            'message': f"Outlier analysis completed for {len(outliers_info)} numeric columns"
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions for distribution-related questions"""
        numeric_df = df.select_dtypes(include=['number'])
        distributions = {}
        
        for col in numeric_df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                distributions[col] = {
                    "mean": data.mean(),
                    "median": data.median(),
                    "std": data.std(),
                    "min": data.min(),
                    "max": data.max(),
                    "skewness": data.skew(),
                    "kurtosis": data.kurtosis()
                }
        
        return distributions
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data for missing data questions"""
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }
        
        return {
            'missing_by_column': missing_info,
            'total_missing': sum(info['count'] for info in missing_info.values()),
            'message': f"Missing data analysis: {len(missing_info)} columns have missing values"
        }
    
    def _general_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """General analysis for other questions"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'message': "General dataset analysis completed"
        }
    
    def generate_visualization_code(self, df: pd.DataFrame, question: str) -> str:
        """Generate Python code for creating visualizations based on the question"""
        question_lower = question.lower()
        
        # Detect visualization intent
        viz_intent = self.detect_visualization_intent(question_lower)
        
        if not viz_intent['chart_type']:
            return ""
        
        chart_type = viz_intent['chart_type']
        columns = viz_intent.get('columns', [])
        
        # Generate appropriate Python code based on chart type
        if chart_type == 'correlation':
            return self._generate_correlation_code(df)
        elif chart_type == 'histogram':
            column = columns[0] if columns else self._get_best_numeric_column(df)
            return self._generate_histogram_code(df, column)
        elif chart_type == 'scatter':
            if len(columns) >= 2:
                return self._generate_scatter_code(df, columns[0], columns[1])
            else:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    return self._generate_scatter_code(df, numeric_cols[0], numeric_cols[1])
        elif chart_type == 'bar':
            column = columns[0] if columns else self._get_best_categorical_column(df)
            return self._generate_bar_code(df, column)
        elif chart_type == 'box':
            column = columns[0] if columns else self._get_best_numeric_column(df)
            return self._generate_box_code(df, column)
        elif chart_type == 'auto':
            # Auto-select best chart type
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0 and any(word in question_lower for word in ['distribution', 'spread', 'pattern']):
                return self._generate_histogram_code(df, numeric_cols[0])
            elif len(numeric_cols) >= 2 and any(word in question_lower for word in ['relationship', 'compare', 'vs']):
                return self._generate_scatter_code(df, numeric_cols[0], numeric_cols[1])
            elif len(categorical_cols) > 0 and any(word in question_lower for word in ['category', 'count', 'frequency']):
                return self._generate_bar_code(df, categorical_cols[0])
            elif len(numeric_cols) >= 2:
                return self._generate_correlation_code(df)
            elif len(numeric_cols) > 0:
                return self._generate_histogram_code(df, numeric_cols[0])
            elif len(categorical_cols) > 0:
                return self._generate_bar_code(df, categorical_cols[0])
        
        return ""
    
    def detect_visualization_intent(self, question: str) -> Dict[str, Any]:
        """Detect what type of visualization the user wants"""
        viz_keywords = {
            'correlation': ['correlation', 'correlate', 'relationship', 'relate', 'heatmap'],
            'histogram': ['histogram', 'distribution', 'spread', 'frequency'],
            'scatter': ['scatter', 'plot', 'relationship between', 'vs', 'against'],
            'bar': ['bar chart', 'bar graph', 'count', 'frequency', 'categories'],
            'box': ['box plot', 'boxplot', 'outliers', 'quartiles'],
            'line': ['line chart', 'trend', 'over time', 'time series']
        }
        
        # Extract potential column names from question
        columns = []
        words = question.split()
        
        for chart_type, keywords in viz_keywords.items():
            if any(keyword in question for keyword in keywords):
                return {
                    'chart_type': chart_type,
                    'columns': columns
                }
        
        # Check for general visualization requests
        if any(word in question for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
            return {
                'chart_type': 'auto',  # Auto-select best chart type
                'columns': columns
            }
        
        return {'chart_type': None, 'columns': []}
    
    def _generate_correlation_code(self, df: pd.DataFrame) -> str:
        """Generate Python code for correlation heatmap"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return ""
        
        code = f"""
# Correlation Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Select numeric columns
numeric_df = df[{numeric_cols}]

# Create correlation matrix
correlation_matrix = numeric_df.corr()

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           square=True, fmt='.2f', cbar_kws={{'shrink': 0.8}})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()

# Print correlation insights
print("\\nCorrelation Analysis:")
strong_corrs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            strong_corrs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

if strong_corrs:
    print(f"Strong correlations found: {{len(strong_corrs)}} pairs with |r| > 0.7")
    for col1, col2, corr_val in strong_corrs[:3]:
        direction = "positive" if corr_val > 0 else "negative"
        print(f"• {{col1}} ↔ {{col2}}: {{direction}} correlation ({{corr_val:.3f}})")
else:
    print("No strong correlations (>0.7) found between variables.")
"""
        return code.strip()
    
    def _generate_histogram_code(self, df: pd.DataFrame, column: str) -> str:
        """Generate Python code for histogram"""
        if column not in df.columns:
            return ""
        
        code = f"""
# Histogram Analysis for {column}
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare data
data = df['{column}'].dropna()

# Create histogram with enhanced styling
plt.figure(figsize=(12, 8))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
plt.title(f'Distribution Analysis: {column}', fontsize=16, fontweight='bold')
plt.xlabel(f'{column}', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Add statistical lines
mean_val = data.mean()
median_val = data.median()
std_val = data.std()

plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {{mean_val:.2f}}')
plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {{median_val:.2f}}')
plt.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1 Std: {{mean_val + std_val:.2f}}')
plt.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1 Std: {{mean_val - std_val:.2f}}')

plt.legend(fontsize=12, framealpha=0.9)
plt.tight_layout()

# Comprehensive statistical analysis
print(f"\\nStatistical Analysis for {column}:")
print(f"• Count: {{len(data):,}} records")
print(f"• Range: {{data.min():.2f}} to {{data.max():.2f}}")
print(f"• Mean: {{data.mean():.2f}}")
print(f"• Median: {{data.median():.2f}}")
print(f"• Standard Deviation: {{data.std():.2f}}")
print(f"• Skewness: {{data.skew():.3f}}")
print(f"• Kurtosis: {{data.kurtosis():.3f}}")

# Distribution shape analysis
skewness = data.skew()
if skewness > 1:
    print("• Distribution is heavily right-skewed")
elif skewness < -1:
    print("• Distribution is heavily left-skewed")
else:
    print("• Distribution is approximately normal")
"""
        return code.strip()
    
    def _generate_scatter_code(self, df: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Generate Python code for scatter plot"""
        if x_col not in df.columns or y_col not in df.columns:
            return ""
        
        code = f"""
# Scatter Plot: {y_col} vs {x_col}
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare data
clean_data = df[['{x_col}', '{y_col}']].dropna()

# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(clean_data['{x_col}'], clean_data['{y_col}'], alpha=0.6, color='blue', s=50)
plt.xlabel('{x_col}', fontsize=14, fontweight='bold')
plt.ylabel('{y_col}', fontsize=14, fontweight='bold')
plt.title(f'{y_col} vs {x_col}', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(clean_data['{x_col}'], clean_data['{y_col}'], 1)
p = np.poly1d(z)
plt.plot(clean_data['{x_col}'], p(clean_data['{x_col}']), "r--", alpha=0.8, linewidth=2, label='Trend line')
plt.legend(fontsize=12)

plt.tight_layout()

# Calculate and print correlation
correlation = clean_data['{x_col}'].corr(clean_data['{y_col}'])
print(f"\\nRelationship Analysis:")
print(f"• Correlation coefficient: {{correlation:.3f}}")
if abs(correlation) > 0.7:
    direction = "strong positive" if correlation > 0 else "strong negative"
    print(f"• {{direction}} linear relationship")
elif abs(correlation) > 0.3:
    direction = "moderate positive" if correlation > 0 else "moderate negative"
    print(f"• {{direction}} linear relationship")
else:
    print(f"• Weak linear relationship")
"""
        return code.strip()
    
    def _generate_bar_code(self, df: pd.DataFrame, column: str) -> str:
        """Generate Python code for bar chart"""
        if column not in df.columns:
            return ""
        
        code = f"""
# Bar Chart: {column}
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data
value_counts = df['{column}'].value_counts().head(10)

# Create bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(value_counts)), value_counts.values, color='skyblue', edgecolor='black')
plt.xlabel('Categories', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.title(f'Top 10 Categories in {column}', fontsize=16, fontweight='bold')
plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{{int(height)}}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Print category analysis
print(f"\\nCategory Analysis for {column}:")
print(f"• Total categories: {{len(value_counts)}}")
print(f"• Most common: '{{value_counts.index[0]}}' ({{value_counts.iloc[0]}} occurrences)")
print(f"• Least common: '{{value_counts.index[-1]}}' ({{value_counts.iloc[-1]}} occurrences)")

# Diversity analysis
total_count = len(df['{column}'].dropna())
top_3_pct = (value_counts.head(3).sum() / total_count) * 100
print(f"• Top 3 categories represent {{top_3_pct:.1f}}% of all data")
"""
        return code.strip()
    
    def _generate_box_code(self, df: pd.DataFrame, column: str) -> str:
        """Generate Python code for box plot"""
        if column not in df.columns:
            return ""
        
        code = f"""
# Box Plot: {column}
import matplotlib.pyplot as plt
import pandas as pd

# Prepare data
data = df['{column}'].dropna()

# Create box plot
plt.figure(figsize=(10, 8))
plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
plt.ylabel('{column}', fontsize=14, fontweight='bold')
plt.title(f'Box Plot: {column}', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add statistics
stats = data.describe()
plt.text(1.1, stats['75%'], f'Q3: {{stats["75%"]:.2f}}', fontsize=12, va='center')
plt.text(1.1, stats['25%'], f'Q1: {{stats["25%"]:.2f}}', fontsize=12, va='center')
plt.text(1.1, stats['50%'], f'Median: {{stats["50%"]:.2f}}', fontsize=12, va='center')

plt.tight_layout()

# Print box plot insights
print(f"\\nBox Plot Analysis for {column}:")
print(f"• Median: {{data.median():.2f}}")
print(f"• Q1 (25th percentile): {{data.quantile(0.25):.2f}}")
print(f"• Q3 (75th percentile): {{data.quantile(0.75):.2f}}")
print(f"• IQR: {{data.quantile(0.75) - data.quantile(0.25):.2f}}")

# Outlier detection
Q1, Q3 = data.quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
if len(outliers) > 0:
    print(f"• {{len(outliers)}} outliers detected")
    print(f"• Outlier range: {{outliers.min():.2f}} to {{outliers.max():.2f}}")
else:
    print("• No outliers detected")
"""
        return code.strip()
    
    def _get_best_numeric_column(self, df: pd.DataFrame) -> str:
        """Get the best numeric column for visualization"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return None
        
        # Prefer columns with meaningful names
        priority_keywords = ['revenue', 'sales', 'profit', 'price', 'amount', 'value', 'score']
        for keyword in priority_keywords:
            for col in numeric_cols:
                if keyword in col.lower():
                    return col
        
        return numeric_cols[0]
    
    def _get_best_categorical_column(self, df: pd.DataFrame) -> str:
        """Get the best categorical column for visualization"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return None
        
        # Prefer columns with reasonable number of categories
        for col in categorical_cols:
            if 2 <= df[col].nunique() <= 20:
                return col
        
        return categorical_cols[0]
    
    def _execute_visualization_code(self, code: str, df: pd.DataFrame) -> str:
        """Execute visualization code and return base64 encoded image"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import numpy as np
            import io
            import base64
            
            # Create execution environment
            exec_globals = {
                'df': df,
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                'print': lambda *args, **kwargs: None  # Suppress print output
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Capture the current figure
            if plt.get_fignums():
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close('all')  # Close all figures to free memory
                print(f"Successfully generated visualization image ({len(img_base64)} chars)")
                return img_base64
            else:
                print("No figures found after code execution")
                return None
            
        except Exception as e:
            print(f"Error executing visualization code: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        return None
