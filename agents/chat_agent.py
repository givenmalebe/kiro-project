import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from .base_agent import BaseAgent
from typing import Dict, Any, List

class ChatAgent(BaseAgent):
    """Agent specialized in conversational data analysis"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Data Chat Agent"
        self.conversation_history = []
    
    def chat_with_data(self, df: pd.DataFrame, user_question: str) -> str:
        """Chat with the data - answer user questions about the dataset"""
        
        # Analyze the dataframe for context
        analysis = self.analyze_dataframe(df)
        
        # Get basic statistics
        stats_summary = self._get_data_summary(df)
        
        # Create context for the AI
        context_prompt = f"""
        You are a Data Analysis Assistant. Answer the user's question about this dataset.
        
        Dataset Information:
        - Shape: {analysis['shape']} (rows, columns)
        - Columns: {analysis['columns']}
        - Numeric columns: {analysis['numeric_columns']}
        - Categorical columns: {analysis['categorical_columns']}
        - Missing values: {analysis['missing_values']}
        
        Data Summary:
        {stats_summary}
        
        Sample data (first 5 rows):
        {df.head().to_string()}
        
        User Question: {user_question}
        
        Provide a comprehensive answer based on the data. If the question requires calculations, 
        perform them and show the results. Be specific and reference actual data values.
        If you need to make assumptions, state them clearly.
        
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
        
        # Check if user is asking for a visualization
        visualization_result = self.handle_visualization_request(df, user_question)
        if visualization_result:
            response += f"\n\n{visualization_result['explanation']}"
            if 'chart' in visualization_result:
                response += f"\n\n[Chart Generated: {visualization_result['chart_type']}]"
        
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
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "No previous conversation"
        
        context_parts = []
        for i, conv in enumerate(self.conversation_history[-3:], 1):  # Last 3 conversations
            context_parts.append(f"Q{i}: {conv['question']}")
            context_parts.append(f"A{i}: {conv['response'][:200]}...")  # Truncate long responses
        
        return "\n".join(context_parts)
    
    def get_suggested_questions(self, df: pd.DataFrame) -> List[str]:
        """Generate suggested questions based on the dataset"""
        analysis = self.analyze_dataframe(df)
        
        suggestions = [
            "What are the key statistics of this dataset?",
            "Are there any missing values I should be concerned about?",
            "What are the main patterns in the data?"
        ]
        
        # Add column-specific suggestions
        if analysis['numeric_columns']:
            suggestions.extend([
                f"What is the distribution of {analysis['numeric_columns'][0]}?",
                f"Are there any outliers in {analysis['numeric_columns'][0]}?",
                "What are the correlations between numeric variables?"
            ])
        
        if analysis['categorical_columns']:
            suggestions.extend([
                f"What are the most common values in {analysis['categorical_columns'][0]}?",
                f"How is {analysis['categorical_columns'][0]} distributed?"
            ])
        
        if len(analysis['numeric_columns']) > 1:
            suggestions.append(f"What is the relationship between {analysis['numeric_columns'][0]} and {analysis['numeric_columns'][1]}?")
        
        return suggestions[:8]  # Return top 8 suggestions
    
    def analyze_specific_question(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze specific aspects based on the question type"""
        question_lower = question.lower()
        
        # Question type detection and specific analysis
        if any(word in question_lower for word in ['correlation', 'relationship', 'related']):
            return self._analyze_correlations(df)
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'unusual']):
            return self._analyze_outliers(df)
        elif any(word in question_lower for word in ['distribution', 'spread', 'range']):
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
        
        # Find strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_value
                })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strongest_correlations": correlations[:5]
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers for outlier-related questions"""
        numeric_df = df.select_dtypes(include=['number'])
        outliers_info = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | 
                                (numeric_df[col] > upper_bound)][col]
            
            outliers_info[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(numeric_df)) * 100,
                "values": outliers.tolist()[:10]  # Show first 10 outliers
            }
        
        return outliers_info
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions for distribution-related questions"""
        numeric_df = df.select_dtypes(include=['number'])
        distributions = {}
        
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
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
    
    def handle_visualization_request(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Handle requests for data visualizations"""
        question_lower = question.lower()
        
        # Detect visualization intent
        viz_intent = self.detect_visualization_intent(question_lower)
        
        if not viz_intent['chart_type']:
            return None
        
        try:
            chart_type = viz_intent['chart_type']
            columns = viz_intent.get('columns', [])
            
            if chart_type == 'correlation':
                return self.create_correlation_heatmap(df)
            elif chart_type == 'histogram':
                column = columns[0] if columns else self._get_best_numeric_column(df)
                return self.create_histogram(df, column)
            elif chart_type == 'scatter':
                if len(columns) >= 2:
                    return self.create_scatter_plot(df, columns[0], columns[1])
                else:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) >= 2:
                        return self.create_scatter_plot(df, numeric_cols[0], numeric_cols[1])
            elif chart_type == 'bar':
                column = columns[0] if columns else self._get_best_categorical_column(df)
                return self.create_bar_chart(df, column)
            elif chart_type == 'box':
                column = columns[0] if columns else self._get_best_numeric_column(df)
                return self.create_box_plot(df, column)
            elif chart_type == 'line':
                return self.create_line_chart(df, columns)
                
        except Exception as e:
            return {
                'explanation': f"I encountered an error creating the visualization: {str(e)}",
                'chart_type': 'error'
            }
        
        return None
    
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
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a correlation heatmap"""
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            return {
                'explanation': "I need at least 2 numeric columns to create a correlation heatmap.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64 for display
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Find strongest correlations
        strong_corrs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append(f"{correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {corr_val:.2f}")
        
        explanation = "Here's a correlation heatmap showing relationships between numeric variables. "
        if strong_corrs:
            explanation += f"Strong correlations found: {', '.join(strong_corrs[:3])}"
        else:
            explanation += "No strong correlations (>0.5) found between variables."
        
        return {
            'chart': img_base64,
            'chart_type': 'correlation_heatmap',
            'explanation': explanation
        }
    
    def create_histogram(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Create a histogram for a numeric column"""
        if column not in df.columns:
            return {
                'explanation': f"Column '{column}' not found in the dataset.",
                'chart_type': 'error'
            }
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                'explanation': f"Column '{column}' is not numeric. Histograms require numeric data.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(10, 6))
        data = df[column].dropna()
        
        plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column}', fontsize=16, fontweight='bold')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        explanation = f"This histogram shows the distribution of {column}. "
        explanation += f"Mean: {mean_val:.2f}, Median: {median_val:.2f}, "
        explanation += f"Standard Deviation: {data.std():.2f}. "
        
        if data.skew() > 1:
            explanation += "The distribution is right-skewed."
        elif data.skew() < -1:
            explanation += "The distribution is left-skewed."
        else:
            explanation += "The distribution is approximately normal."
        
        return {
            'chart': img_base64,
            'chart_type': 'histogram',
            'explanation': explanation
        }
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """Create a scatter plot between two numeric columns"""
        if x_col not in df.columns or y_col not in df.columns:
            return {
                'explanation': f"One or both columns '{x_col}', '{y_col}' not found.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(10, 6))
        
        # Remove rows with missing values
        clean_data = df[[x_col, y_col]].dropna()
        
        plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, color='blue')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
        p = np.poly1d(z)
        plt.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8, label='Trend line')
        plt.legend()
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Calculate correlation
        correlation = clean_data[x_col].corr(clean_data[y_col])
        
        explanation = f"This scatter plot shows the relationship between {x_col} and {y_col}. "
        explanation += f"Correlation coefficient: {correlation:.3f}. "
        
        if abs(correlation) > 0.7:
            explanation += "Strong correlation detected."
        elif abs(correlation) > 0.3:
            explanation += "Moderate correlation detected."
        else:
            explanation += "Weak or no correlation detected."
        
        return {
            'chart': img_base64,
            'chart_type': 'scatter_plot',
            'explanation': explanation
        }
    
    def create_bar_chart(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Create a bar chart for categorical data"""
        if column not in df.columns:
            return {
                'explanation': f"Column '{column}' not found in the dataset.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(12, 6))
        
        value_counts = df[column].value_counts().head(10)  # Top 10 categories
        
        plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(f'Distribution of {column}', fontsize=16, fontweight='bold')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        explanation = f"This bar chart shows the distribution of {column}. "
        explanation += f"Most common value: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences). "
        explanation += f"Total unique values: {df[column].nunique()}."
        
        return {
            'chart': img_base64,
            'chart_type': 'bar_chart',
            'explanation': explanation
        }
    
    def create_box_plot(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Create a box plot for numeric data"""
        if column not in df.columns:
            return {
                'explanation': f"Column '{column}' not found in the dataset.",
                'chart_type': 'error'
            }
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                'explanation': f"Column '{column}' is not numeric. Box plots require numeric data.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(8, 6))
        
        data = df[column].dropna()
        plt.boxplot(data, vert=True, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        plt.ylabel(column)
        plt.title(f'Box Plot of {column}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Calculate outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        
        explanation = f"This box plot shows the distribution and outliers in {column}. "
        explanation += f"Median: {data.median():.2f}, IQR: {IQR:.2f}. "
        explanation += f"Found {len(outliers)} outliers ({len(outliers)/len(data)*100:.1f}% of data)."
        
        return {
            'chart': img_base64,
            'chart_type': 'box_plot',
            'explanation': explanation
        }
    
    def create_line_chart(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Create a line chart for time series or sequential data"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return {
                'explanation': "No numeric columns found for line chart.",
                'chart_type': 'error'
            }
        
        plt.figure(figsize=(12, 6))
        
        # Use first numeric column or specified column
        y_col = columns[0] if columns and columns[0] in numeric_cols else numeric_cols[0]
        
        # Create x-axis (index or first column)
        x_data = range(len(df))
        y_data = df[y_col].fillna(df[y_col].mean())
        
        plt.plot(x_data, y_data, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Index')
        plt.ylabel(y_col)
        plt.title(f'Trend of {y_col}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        explanation = f"This line chart shows the trend of {y_col} over the dataset sequence. "
        
        # Simple trend analysis
        if len(y_data) > 1:
            slope = (y_data.iloc[-1] - y_data.iloc[0]) / len(y_data)
            if slope > 0:
                explanation += "Overall upward trend detected."
            elif slope < 0:
                explanation += "Overall downward trend detected."
            else:
                explanation += "Relatively stable trend."
        
        return {
            'chart': img_base64,
            'chart_type': 'line_chart',
            'explanation': explanation
        }
    
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
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data for missing data questions"""
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_info[col] = {
                "missing_count": missing_count,
                "missing_percentage": (missing_count / len(df)) * 100
            }
        
        return missing_info
    
    def _general_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """General analysis for other questions"""
        return {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "describe": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {},
            "sample": df.head().to_dict()
        }