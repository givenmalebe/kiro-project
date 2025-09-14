"""
Pattern Recognition System for Automatic Data Analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class PatternRecognitionSystem:
    """Advanced pattern recognition for automatic data analysis"""
    
    def __init__(self):
        self.correlation_threshold = 0.5
        self.significance_level = 0.05
        self.outlier_threshold = 1.5  # IQR multiplier
        
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive correlation analysis with significance testing"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                'correlation_matrix': {},
                'strong_correlations': [],
                'correlation_clusters': [],
                'correlation_summary': 'Insufficient numeric columns for correlation analysis'
            }
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find significant correlations
        strong_correlations = []
        moderate_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_val):
                    # Calculate p-value for correlation significance
                    n = len(df[[col1, col2]].dropna())
                    if n > 3:
                        t_stat = corr_val * np.sqrt((n-2) / (1 - corr_val**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                        
                        correlation_info = {
                            'variable_1': col1,
                            'variable_2': col2,
                            'correlation': corr_val,
                            'p_value': p_value,
                            'significant': p_value < self.significance_level,
                            'strength': self._classify_correlation_strength(abs(corr_val)),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        }
                        
                        if abs(corr_val) >= 0.7:
                            strong_correlations.append(correlation_info)
                        elif abs(corr_val) >= self.correlation_threshold:
                            moderate_correlations.append(correlation_info)
        
        # Sort by correlation strength
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        moderate_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Identify correlation clusters
        correlation_clusters = self._identify_correlation_clusters(corr_matrix)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'moderate_correlations': moderate_correlations,
            'correlation_clusters': correlation_clusters,
            'correlation_summary': self._generate_correlation_summary(strong_correlations, moderate_correlations)
        }
    
    def detect_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect trends in time series and sequential data"""
        trends = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 5:  # Need minimum data points for trend analysis
                continue
            
            # Linear trend analysis
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
            
            # Classify trend strength and direction
            trend_strength = abs(r_value)
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_significance = p_value < self.significance_level
            
            # Detect seasonality (if enough data points)
            seasonality_info = None
            if len(data) >= 24:  # Need sufficient data for seasonality
                seasonality_info = self._detect_seasonality(data.values)
            
            # Detect change points
            change_points = self._detect_change_points(data.values)
            
            trend_info = {
                'column': col,
                'trend_direction': trend_direction,
                'trend_strength': self._classify_trend_strength(trend_strength),
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significant': trend_significance,
                'seasonality': seasonality_info,
                'change_points': change_points,
                'volatility': data.std() / data.mean() if data.mean() != 0 else 0
            }
            
            trends.append(trend_info)
        
        return sorted(trends, key=lambda x: x['r_squared'], reverse=True)
    
    def find_anomalies(self, df: pd.DataFrame) -> Dict[str, List[Any]]:
        """Comprehensive anomaly detection using multiple methods"""
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 4:
                continue
            
            col_anomalies = {
                'outliers_iqr': [],
                'outliers_zscore': [],
                'outliers_modified_zscore': [],
                'statistical_anomalies': []
            }
            
            # IQR-based outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            col_anomalies['outliers_iqr'] = {
                'indices': iqr_outliers.index.tolist(),
                'values': iqr_outliers.values.tolist(),
                'count': len(iqr_outliers),
                'percentage': len(iqr_outliers) / len(data) * 100
            }
            
            # Z-score based outliers
            z_scores = np.abs(stats.zscore(data))
            zscore_outliers = data[z_scores > 3]
            col_anomalies['outliers_zscore'] = {
                'indices': zscore_outliers.index.tolist(),
                'values': zscore_outliers.values.tolist(),
                'count': len(zscore_outliers),
                'percentage': len(zscore_outliers) / len(data) * 100
            }
            
            # Modified Z-score (more robust)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            mod_zscore_outliers = data[np.abs(modified_z_scores) > 3.5]
            col_anomalies['outliers_modified_zscore'] = {
                'indices': mod_zscore_outliers.index.tolist(),
                'values': mod_zscore_outliers.values.tolist(),
                'count': len(mod_zscore_outliers),
                'percentage': len(mod_zscore_outliers) / len(data) * 100
            }
            
            # Statistical anomalies (extreme values)
            extreme_low = data[data < data.quantile(0.01)]
            extreme_high = data[data > data.quantile(0.99)]
            col_anomalies['statistical_anomalies'] = {
                'extreme_low': {
                    'indices': extreme_low.index.tolist(),
                    'values': extreme_low.values.tolist(),
                    'count': len(extreme_low)
                },
                'extreme_high': {
                    'indices': extreme_high.index.tolist(),
                    'values': extreme_high.values.tolist(),
                    'count': len(extreme_high)
                }
            }
            
            anomalies[col] = col_anomalies
        
        return anomalies
    
    def classify_business_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Classify business context and domain from data characteristics"""
        column_names = [col.lower() for col in df.columns]
        
        # Business domain indicators
        domain_indicators = {
            'ecommerce': {
                'keywords': ['order', 'product', 'customer', 'price', 'quantity', 'cart', 'purchase', 'sku'],
                'patterns': ['order_id', 'product_name', 'customer_id', 'unit_price']
            },
            'finance': {
                'keywords': ['revenue', 'profit', 'cost', 'investment', 'return', 'balance', 'amount', 'portfolio'],
                'patterns': ['total_amount', 'net_profit', 'gross_revenue', 'cost_center']
            },
            'marketing': {
                'keywords': ['campaign', 'conversion', 'click', 'impression', 'lead', 'acquisition', 'ctr', 'roi'],
                'patterns': ['campaign_id', 'click_rate', 'conversion_rate', 'cost_per_click']
            },
            'sales': {
                'keywords': ['deal', 'opportunity', 'pipeline', 'quota', 'commission', 'territory', 'lead'],
                'patterns': ['deal_value', 'sales_rep', 'close_date', 'pipeline_stage']
            },
            'hr': {
                'keywords': ['employee', 'salary', 'department', 'performance', 'training', 'retention'],
                'patterns': ['employee_id', 'hire_date', 'department_name', 'performance_rating']
            },
            'operations': {
                'keywords': ['production', 'inventory', 'supply', 'logistics', 'efficiency', 'quality'],
                'patterns': ['production_date', 'inventory_level', 'quality_score', 'efficiency_rate']
            }
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            keyword_score = sum(1 for keyword in indicators['keywords'] 
                              if any(keyword in col for col in column_names))
            pattern_score = sum(1 for pattern in indicators['patterns']
                              if any(pattern in col for col in column_names))
            
            total_score = keyword_score + (pattern_score * 2)  # Weight patterns higher
            if total_score > 0:
                domain_scores[domain] = total_score
        
        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain] / sum(domain_scores.values())
        else:
            primary_domain = 'general'
            confidence = 0.0
        
        # Analyze data characteristics
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        categorical_ratio = len(df.select_dtypes(include=['object']).columns) / len(df.columns)
        
        return {
            'primary_domain': primary_domain,
            'confidence': confidence,
            'domain_scores': domain_scores,
            'data_characteristics': {
                'numeric_ratio': numeric_ratio,
                'categorical_ratio': categorical_ratio,
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'missing_data_ratio': df.isnull().sum().sum() / df.size
            },
            'recommended_analysis': self._recommend_analysis_type(primary_domain, numeric_ratio, categorical_ratio)
        }
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength"""
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
    
    def _classify_trend_strength(self, r_value: float) -> str:
        """Classify trend strength based on R-value"""
        if r_value >= 0.8:
            return 'very_strong'
        elif r_value >= 0.6:
            return 'strong'
        elif r_value >= 0.4:
            return 'moderate'
        elif r_value >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _identify_correlation_clusters(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify clusters of highly correlated variables"""
        if len(corr_matrix.columns) < 3:
            return []
        
        try:
            # Use hierarchical clustering on correlation matrix
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix.values)
            np.fill_diagonal(distance_matrix, 0)
            
            # Perform clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='average')
            
            # Get clusters
            clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            # Group variables by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(corr_matrix.columns[i])
            
            # Format cluster information
            cluster_info = []
            for cluster_id, variables in cluster_groups.items():
                if len(variables) > 1:  # Only include clusters with multiple variables
                    # Calculate average correlation within cluster
                    cluster_corrs = []
                    for i in range(len(variables)):
                        for j in range(i+1, len(variables)):
                            corr_val = corr_matrix.loc[variables[i], variables[j]]
                            if not np.isnan(corr_val):
                                cluster_corrs.append(abs(corr_val))
                    
                    avg_correlation = np.mean(cluster_corrs) if cluster_corrs else 0
                    
                    cluster_info.append({
                        'cluster_id': int(cluster_id),
                        'variables': variables,
                        'size': len(variables),
                        'average_correlation': avg_correlation
                    })
            
            return sorted(cluster_info, key=lambda x: x['average_correlation'], reverse=True)
        
        except ImportError:
            return []
        except Exception:
            return []
    
    def _detect_seasonality(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect seasonality in time series data"""
        try:
            from scipy.fft import fft, fftfreq
            
            # Remove trend
            detrended = data - np.linspace(data[0], data[-1], len(data))
            
            # Apply FFT
            fft_values = fft(detrended)
            frequencies = fftfreq(len(data))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values)**2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_frequency = frequencies[dominant_freq_idx]
            
            if dominant_frequency > 0:
                period = 1 / dominant_frequency
                strength = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
                
                return {
                    'detected': True,
                    'period': period,
                    'strength': strength,
                    'dominant_frequency': dominant_frequency
                }
        except:
            pass
        
        return {'detected': False}
    
    def _detect_change_points(self, data: np.ndarray) -> List[int]:
        """Detect change points in time series data"""
        if len(data) < 10:
            return []
        
        try:
            # Simple change point detection using variance
            window_size = max(5, len(data) // 10)
            change_points = []
            
            for i in range(window_size, len(data) - window_size):
                left_window = data[i-window_size:i]
                right_window = data[i:i+window_size]
                
                # Test for significant difference in means
                t_stat, p_value = stats.ttest_ind(left_window, right_window)
                
                if p_value < 0.01:  # Significant change
                    change_points.append(i)
            
            # Remove nearby change points (keep only significant ones)
            filtered_change_points = []
            for cp in change_points:
                if not filtered_change_points or cp - filtered_change_points[-1] > window_size:
                    filtered_change_points.append(cp)
            
            return filtered_change_points
        except:
            return []
    
    def _generate_correlation_summary(self, strong_corrs: List[Dict], moderate_corrs: List[Dict]) -> str:
        """Generate a summary of correlation findings"""
        total_strong = len(strong_corrs)
        total_moderate = len(moderate_corrs)
        
        if total_strong == 0 and total_moderate == 0:
            return "No significant correlations found in the data."
        
        summary_parts = []
        
        if total_strong > 0:
            summary_parts.append(f"Found {total_strong} strong correlation{'s' if total_strong != 1 else ''}")
            
        if total_moderate > 0:
            summary_parts.append(f"Found {total_moderate} moderate correlation{'s' if total_moderate != 1 else ''}")
        
        # Highlight top correlation
        if strong_corrs:
            top_corr = strong_corrs[0]
            summary_parts.append(f"Strongest: {top_corr['variable_1']} ↔ {top_corr['variable_2']} (r={top_corr['correlation']:.3f})")
        elif moderate_corrs:
            top_corr = moderate_corrs[0]
            summary_parts.append(f"Strongest: {top_corr['variable_1']} ↔ {top_corr['variable_2']} (r={top_corr['correlation']:.3f})")
        
        return ". ".join(summary_parts) + "."
    
    def _recommend_analysis_type(self, domain: str, numeric_ratio: float, categorical_ratio: float) -> List[str]:
        """Recommend analysis types based on domain and data characteristics"""
        recommendations = []
        
        # Domain-specific recommendations
        domain_recommendations = {
            'finance': ['financial_performance', 'trend_analysis', 'correlation_heatmap'],
            'sales': ['business_kpi_dashboard', 'trend_analysis', 'category_performance'],
            'marketing': ['conversion_analysis', 'campaign_performance', 'correlation_heatmap'],
            'ecommerce': ['product_analysis', 'customer_segmentation', 'revenue_trends'],
            'operations': ['efficiency_analysis', 'quality_metrics', 'process_optimization'],
            'hr': ['performance_analysis', 'retention_analysis', 'compensation_analysis']
        }
        
        recommendations.extend(domain_recommendations.get(domain, ['business_kpi_dashboard']))
        
        # Data characteristic recommendations
        if numeric_ratio > 0.7:
            recommendations.extend(['correlation_heatmap', 'trend_analysis'])
        
        if categorical_ratio > 0.3:
            recommendations.extend(['category_analysis', 'segmentation_analysis'])
        
        if numeric_ratio > 0.5 and categorical_ratio > 0.2:
            recommendations.append('mixed_analysis_dashboard')
        
        return list(set(recommendations))  # Remove duplicates