import pandas as pd
import numpy as np
from scipy import stats
from .base_agent import BaseAgent
from typing import Dict, Any, List

class DescriptiveAgent(BaseAgent):
    """Agent specialized in Descriptive Analysis"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Descriptive Analysis Agent"
    
    def perform_descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive descriptive analysis"""
        analysis = self.analyze_dataframe(df)
        
        # Generate descriptive insights using AI
        descriptive_prompt = f"""
        As a Statistical Analyst, provide descriptive analysis for this dataset:
        
        Dataset Info:
        - Shape: {analysis['shape']}
        - Numeric columns: {analysis['numeric_columns']}
        - Categorical columns: {analysis['categorical_columns']}
        
        Statistical Summary:
        {df.describe().to_string()}
        
        Provide insights about:
        1. Central tendencies and variability
        2. Distribution shapes and patterns
        3. Data concentration and spread
        4. Categorical variable distributions
        5. Key statistical findings
        6. Business implications of the statistics
        
        Be detailed and explain what these statistics mean for business decisions.
        """
        
        ai_insights = self.generate_response(descriptive_prompt)
        
        return {
            "ai_insights": ai_insights,
            "central_tendency": self._analyze_central_tendency(df),
            "variability_measures": self._analyze_variability(df),
            "distribution_analysis": self._analyze_distributions(df),
            "categorical_analysis": self._analyze_categorical_variables(df),
            "percentile_analysis": self._analyze_percentiles(df)
        }
    
    def _analyze_central_tendency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze measures of central tendency"""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        central_tendency = {}
        for col in numeric_df.columns:
            central_tendency[col] = {
                "mean": numeric_df[col].mean(),
                "median": numeric_df[col].median(),
                "mode": numeric_df[col].mode().iloc[0] if not numeric_df[col].mode().empty else None,
                "geometric_mean": stats.gmean(numeric_df[col].dropna()) if (numeric_df[col] > 0).all() else None,
                "harmonic_mean": stats.hmean(numeric_df[col].dropna()) if (numeric_df[col] > 0).all() else None
            }
        
        return central_tendency
    
    def _analyze_variability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze measures of variability"""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        variability = {}
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            variability[col] = {
                "variance": data.var(),
                "standard_deviation": data.std(),
                "range": data.max() - data.min(),
                "interquartile_range": data.quantile(0.75) - data.quantile(0.25),
                "coefficient_of_variation": (data.std() / data.mean()) * 100 if data.mean() != 0 else None,
                "mean_absolute_deviation": np.mean(np.abs(data - data.mean()))
            }
        
        return variability
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution characteristics"""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        distributions = {}
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            distributions[col] = {
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "normality_test": {
                    "shapiro_wilk": stats.shapiro(data.sample(min(5000, len(data)))) if len(data) > 3 else None,
                    "jarque_bera": stats.jarque_bera(data) if len(data) > 2 else None
                },
                "distribution_type": self._classify_distribution(data)
            }
        
        return distributions
    
    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5:
            skew_type = "approximately symmetric"
        elif skewness > 0.5:
            skew_type = "positively skewed (right-tailed)"
        else:
            skew_type = "negatively skewed (left-tailed)"
        
        if kurtosis < -1:
            kurt_type = "platykurtic (flatter than normal)"
        elif kurtosis > 1:
            kurt_type = "leptokurtic (more peaked than normal)"
        else:
            kurt_type = "mesokurtic (similar to normal)"
        
        return f"{skew_type}, {kurt_type}"
    
    def _analyze_categorical_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables"""
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if categorical_df.empty:
            return {"message": "No categorical columns found"}
        
        categorical_analysis = {}
        for col in categorical_df.columns:
            value_counts = categorical_df[col].value_counts()
            categorical_analysis[col] = {
                "unique_values": categorical_df[col].nunique(),
                "most_frequent": value_counts.index[0],
                "most_frequent_count": value_counts.iloc[0],
                "least_frequent": value_counts.index[-1],
                "least_frequent_count": value_counts.iloc[-1],
                "frequency_distribution": value_counts.head(10).to_dict(),
                "entropy": stats.entropy(value_counts.values),
                "concentration_ratio": (value_counts.head(3).sum() / value_counts.sum()) * 100
            }
        
        return categorical_analysis
    
    def _analyze_percentiles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze percentile distributions"""
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return {"message": "No numeric columns found"}
        
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        percentile_analysis = {}
        
        for col in numeric_df.columns:
            percentile_values = {}
            for p in percentiles:
                percentile_values[f"p{p}"] = numeric_df[col].quantile(p/100)
            
            percentile_analysis[col] = {
                "percentiles": percentile_values,
                "outlier_boundaries": {
                    "mild_outliers_lower": numeric_df[col].quantile(0.25) - 1.5 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)),
                    "mild_outliers_upper": numeric_df[col].quantile(0.75) + 1.5 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)),
                    "extreme_outliers_lower": numeric_df[col].quantile(0.25) - 3 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)),
                    "extreme_outliers_upper": numeric_df[col].quantile(0.75) + 3 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25))
                }
            }
        
        return percentile_analysis