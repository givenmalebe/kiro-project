import pandas as pd
from typing import Dict, List, Any, Tuple

class DataValidator:
    """Utility class for validating uploaded data"""
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate uploaded CSV data
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues
        
        # Check minimum size requirements
        if len(df) < 2:
            issues.append("Dataset must have at least 2 rows")
        
        if len(df.columns) < 1:
            issues.append("Dataset must have at least 1 column")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Completely empty columns found: {empty_cols}")
        
        # Check data types
        if df.select_dtypes(include=['number']).empty and df.select_dtypes(include=['object']).empty:
            issues.append("No recognizable data types found")
        
        # Check for excessive missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 90:
            issues.append(f"Excessive missing data: {missing_percentage:.1f}%")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names found")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get a comprehensive summary of the dataset"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    @staticmethod
    def suggest_data_improvements(df: pd.DataFrame) -> List[str]:
        """Suggest improvements for the dataset"""
        suggestions = []
        
        # Missing data suggestions
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing_cols = missing_data[missing_data > len(df) * 0.5].index.tolist()
            if high_missing_cols:
                suggestions.append(f"Consider removing columns with >50% missing data: {high_missing_cols}")
        
        # Data type suggestions
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Check if object column might be numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                non_null_after_conversion = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if non_null_after_conversion > len(df) * 0.8:
                    suggestions.append(f"Column '{col}' might be converted to numeric")
            except:
                pass
            
            # Check if object column might be datetime
            try:
                pd.to_datetime(df[col], errors='coerce')
                non_null_after_conversion = pd.to_datetime(df[col], errors='coerce').notna().sum()
                if non_null_after_conversion > len(df) * 0.8:
                    suggestions.append(f"Column '{col}' might be converted to datetime")
            except:
                pass
        
        # Duplicate data suggestions
        if df.duplicated().sum() > 0:
            suggestions.append(f"Found {df.duplicated().sum()} duplicate rows - consider removing")
        
        return suggestions