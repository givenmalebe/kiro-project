import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from .base_agent import BaseAgent
from typing import Dict, Any, List

class EDAAgent(BaseAgent):
    """Agent specialized in Exploratory Data Analysis"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Exploratory Data Analysis Agent"
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive exploratory data analysis"""
        analysis = self.analyze_dataframe(df)
        
        # Generate EDA insights using AI
        eda_prompt = f"""
        As a Data Scientist, perform exploratory data analysis on this dataset:
        
        Dataset Overview:
        - Shape: {analysis['shape']}
        - Columns: {analysis['columns']}
        - Data types: {analysis['dtypes']}
        - Missing values: {analysis['missing_values']}
        
        Sample data:
        {df.head(10).to_string()}
        
        Statistical summary:
        {df.describe().to_string()}
        
        Provide detailed insights about:
        1. Data quality issues
        2. Patterns and anomalies
        3. Relationships between variables
        4. Recommendations for data preprocessing
        5. Interesting findings
        
        Be specific and actionable.
        """
        
        ai_insights = self.generate_response(eda_prompt)
        
        return {
            "ai_insights": ai_insights,
            "data_overview": self._get_data_overview(df),
            "statistical_summary": self._get_statistical_summary(df),
            "missing_data_analysis": self._analyze_missing_data(df),
            "correlation_analysis": self._analyze_correlations(df),
            "outlier_analysis": self._detect_outliers(df)
        }
    
    def _get_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data overview"""
        return {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
            "column_info": {
                col: {
                    "dtype": str(df[col].dtype),
                    "unique_values": df[col].nunique(),
                    "null_count": df[col].isnull().sum(),
                    "null_percentage": (df[col].isnull().sum() / len(df)) * 100
                }
                for col in df.columns
            }
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary for numeric columns"""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        return {
            "describe": numeric_df.describe().to_dict(),
            "skewness": numeric_df.skew().to_dict(),
            "kurtosis": numeric_df.kurtosis().to_dict()
        }
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": missing_percentage.to_dict(),
            "columns_with_missing": missing_data[missing_data > 0].index.tolist(),
            "total_missing": missing_data.sum(),
            "missing_data_pattern": df.isnull().sum().sum()
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numeric columns for correlation analysis"}
        
        correlation_matrix = numeric_df.corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        "var1": correlation_matrix.columns[i],
                        "var2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": high_corr_pairs
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
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
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        
        return outliers_info