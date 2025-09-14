import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from .base_agent import BaseAgent
from typing import Dict, Any, List, Tuple
import streamlit as st

class PrescriptiveAgent(BaseAgent):
    """Agent specialized in Prescriptive Analysis and Recommendations"""
    
    def __init__(self):
        super().__init__()
        self.agent_name = "Prescriptive Analysis Agent"
    
    def perform_prescriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive prescriptive analysis with actionable recommendations"""
        import time
        start_time = time.time()
        
        # Use full dataset for comprehensive analysis
        df_sample = df
        try:
            st.info(f"📊 Analyzing complete dataset of {len(df):,} records for comprehensive analysis")
        except:
            print(f"📊 Analyzing complete dataset of {len(df):,} records for comprehensive analysis")
        
        analysis = self.analyze_dataframe(df_sample)
        
        # Comprehensive business domain detection
        business_domain = self._detect_business_domain(df_sample)
        
        # Full performance drivers identification
        performance_drivers = self._identify_performance_drivers(df_sample, analysis)
        
        # Generate comprehensive strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(df_sample, analysis, business_domain)
        
        # Full predictive modeling
        predictive_insights = self._perform_predictive_modeling(df_sample, analysis)
        
        # Generate comprehensive AI insights
        prescriptive_prompt = f"""
        Comprehensive Business Analysis for {business_domain} data ({analysis['shape'][0]} records):
        
        Key Metrics: {analysis['numeric_columns'][:5]}
        Categories: {analysis['categorical_columns'][:3]}
        
        Provide detailed recommendations covering:
        1. **Revenue Growth**: Specific strategies to increase performance
        2. **Risk Management**: Key risks and mitigation strategies
        3. **Operational Efficiency**: Process improvements and optimization
        4. **Market Opportunities**: Growth areas and expansion possibilities
        5. **Strategic Initiatives**: Long-term strategic recommendations
        
        Make recommendations specific, actionable, and data-driven.
        """
        
        ai_insights = self.generate_response(prescriptive_prompt)
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        
        return {
            "ai_insights": ai_insights,
            "business_domain": business_domain,
            "performance_drivers": performance_drivers,
            "strategic_recommendations": strategic_recommendations,
            "predictive_insights": predictive_insights,
            "optimization_opportunities": self._identify_optimization_opportunities(df_sample, analysis),
            "predictive_models": self._build_predictive_models(df_sample),
            "recommendations": self._generate_recommendations(df_sample),
            "risk_analysis": self._perform_risk_analysis(df_sample),
            "scenario_analysis": self._perform_scenario_analysis(df_sample),
            "processing_time": elapsed_time,
            "analysis_type": "comprehensive"
        }
    
    def _identify_optimization_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify optimization opportunities in the data"""
        opportunities = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Identify columns with high variability (optimization potential)
        for col in numeric_cols:
            cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
            if cv > 50:  # High coefficient of variation
                opportunities[f"{col}_variability"] = {
                    "type": "High Variability",
                    "coefficient_of_variation": cv,
                    "recommendation": f"High variability in {col} suggests optimization potential"
                }
        
        # Identify outliers as optimization targets
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                opportunities[f"{col}_outliers"] = {
                    "type": "Outlier Management",
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(df)) * 100,
                    "recommendation": f"Address {len(outliers)} outliers in {col} for process improvement"
                }
        
        return opportunities
    
    def _build_predictive_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build simple predictive models for key variables"""
        models = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Need at least 2 numeric columns for predictive modeling"}
        
        # Try to build models for each numeric column
        for target_col in numeric_cols[:3]:  # Limit to first 3 columns
            try:
                # Prepare features (all other numeric columns)
                feature_cols = [col for col in numeric_cols if col != target_col]
                if not feature_cols:
                    continue
                
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = df[target_col].fillna(df[target_col].mean())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Build model (balanced for comprehensive analysis)
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                # Feature importance
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
                
                models[target_col] = {
                    "model_type": "Random Forest Regression",
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "feature_importance": feature_importance,
                    "top_features": sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:3]
                }
                
            except Exception as e:
                models[target_col] = {"error": str(e)}
        
        return models
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate specific business recommendations"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Data quality recommendations
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            recommendations.append({
                "category": "Data Quality",
                "priority": "High",
                "recommendation": "Address missing data issues",
                "details": f"Found {missing_data.sum()} missing values across {(missing_data > 0).sum()} columns",
                "action_items": [
                    "Implement data validation processes",
                    "Set up automated data quality monitoring",
                    "Create data imputation strategies"
                ]
            })
        
        # Performance optimization recommendations
        for col in numeric_cols:
            if df[col].std() > 0:
                cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
                if cv > 30:  # Moderate to high variability (lowered threshold)
                    priority = "High" if cv > 80 else "Medium"
                    recommendations.append({
                        "category": "Performance Optimization",
                        "priority": priority,
                        "recommendation": f"Reduce variability in {col}",
                        "details": f"Coefficient of variation: {cv:.2f}%",
                        "action_items": [
                            f"Investigate root causes of {col} variability",
                            "Implement process standardization",
                            "Set up monitoring and control limits"
                        ]
                    })
        
        # Categorical analysis recommendations
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:  # Moderate to high cardinality (lowered threshold)
                priority = "Medium" if unique_ratio > 0.8 else "Low"
                recommendations.append({
                    "category": "Data Management",
                    "priority": priority,
                    "recommendation": f"Consider grouping categories in {col}",
                    "details": f"High cardinality: {df[col].nunique()} unique values ({unique_ratio:.1%} unique)",
                    "action_items": [
                        "Group low-frequency categories",
                        "Create hierarchical categorization",
                        "Implement category management system"
                    ]
                })
        
        # Add general business improvement recommendations
        if len(recommendations) < 3:  # Ensure we always have some recommendations
            # Revenue/profit optimization
            revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                           for keyword in ['revenue', 'sales', 'profit', 'income', 'earnings'])]
            if revenue_cols:
                col = revenue_cols[0]
                q75 = df[col].quantile(0.75)
                below_target = len(df[df[col] < q75])
                if below_target > 0:
                    recommendations.append({
                        "category": "Revenue Growth",
                        "priority": "High",
                        "recommendation": f"Improve performance of bottom quartile in {col}",
                        "details": f"{below_target} records ({below_target/len(df)*100:.1f}%) below 75th percentile",
                        "action_items": [
                            f"Analyze top performers in {col}",
                            "Identify success factors and best practices",
                            "Implement improvement programs"
                        ]
                    })
            
            # Process improvement recommendation
            if len(numeric_cols) >= 2:
                recommendations.append({
                    "category": "Process Improvement",
                    "priority": "Medium",
                    "recommendation": "Implement data-driven decision making",
                    "details": f"Leverage {len(numeric_cols)} numeric metrics for optimization",
                    "action_items": [
                        "Set up regular performance monitoring",
                        "Create automated reporting dashboards",
                        "Establish KPI targets and tracking"
                    ]
                })
        
        return recommendations
    
    def _perform_risk_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform risk analysis on the data"""
        risks = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Calculate risk metrics
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            # Value at Risk (VaR) - 5th percentile
            var_5 = data.quantile(0.05)
            var_1 = data.quantile(0.01)
            
            # Expected Shortfall (average of worst 5%)
            worst_5_percent = data[data <= var_5]
            expected_shortfall = worst_5_percent.mean() if len(worst_5_percent) > 0 else var_5
            
            risks[col] = {
                "value_at_risk_5%": var_5,
                "value_at_risk_1%": var_1,
                "expected_shortfall": expected_shortfall,
                "volatility": data.std(),
                "risk_level": self._classify_risk_level(data.std(), data.mean())
            }
        
        return risks
    
    def _classify_risk_level(self, std: float, mean: float) -> str:
        """Classify risk level based on coefficient of variation"""
        if mean == 0:
            return "Undefined"
        
        cv = (std / abs(mean)) * 100
        
        if cv < 15:
            return "Low Risk"
        elif cv < 30:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _perform_scenario_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform scenario analysis"""
        scenarios = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            data = df[col].dropna()
            if len(data) == 0:
                continue
            
            mean_val = data.mean()
            std_val = data.std()
            
            scenarios[col] = {
                "optimistic": mean_val + std_val,
                "realistic": mean_val,
                "pessimistic": mean_val - std_val,
                "best_case": data.quantile(0.95),
                "worst_case": data.quantile(0.05),
                "scenario_range": data.quantile(0.95) - data.quantile(0.05)
            }
        
        return scenarios
    
    def _identify_optimization_opportunities_fast(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Fast optimization opportunities identification"""
        opportunities = {}
        numeric_cols = analysis['numeric_columns'][:3]  # Limit to 3 columns
        
        for col in numeric_cols:
            cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
            if cv > 50:  # High variability
                opportunities[f"{col}_variability"] = {
                    "type": "High Variability",
                    "coefficient_of_variation": cv,
                    "recommendation": f"Reduce variability in {col}"
                }
        
        return opportunities
    
    def _generate_recommendations_fast(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate fast recommendations with limited analysis"""
        recommendations = []
        
        # Quick data quality check
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            recommendations.append({
                "category": "Data Quality",
                "priority": "High",
                "recommendation": "Address missing data issues",
                "details": f"Found {missing_data} missing values"
            })
        
        # Quick performance check on first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
            if cv > 100:
                recommendations.append({
                    "category": "Performance",
                    "priority": "Medium",
                    "recommendation": f"Reduce variability in {col}",
                    "details": f"High variation: {cv:.1f}%"
                })
        
        return recommendations[:3]  # Max 3 recommendations
    
    def _detect_business_domain_fast(self, df: pd.DataFrame) -> str:
        """Fast business domain detection using simplified keyword matching"""
        domain_indicators = {
            'ecommerce': ['order', 'product', 'price', 'quantity'],
            'finance': ['revenue', 'profit', 'cost', 'investment'],
            'marketing': ['campaign', 'conversion', 'click', 'lead'],
            'sales': ['deal', 'opportunity', 'pipeline', 'quota'],
            'hr': ['employee', 'salary', 'department', 'performance'],
            'operations': ['production', 'inventory', 'supply', 'efficiency']
        }
        
        column_names = [col.lower() for col in df.columns]
        
        for domain, keywords in domain_indicators.items():
            if any(keyword in col for keyword in keywords for col in column_names):
                return domain
        return 'general business'
    
    def _detect_business_domain(self, df: pd.DataFrame) -> str:
        """Detect business domain from column names and data patterns"""
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
            return max(domain_scores, key=domain_scores.get)
        return 'general business'
    
    def _identify_performance_drivers_fast(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Fast performance drivers identification with limited analysis"""
        numeric_cols = analysis['numeric_columns'][:5]  # Limit to first 5 columns
        
        if len(numeric_cols) < 2:
            return {"summary": "Insufficient numeric data", "drivers": []}
        
        # Quick target variable detection
        target_keywords = ['revenue', 'sales', 'profit', 'amount']
        target_col = None
        
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0]
        
        drivers = []
        
        # Quick correlation analysis (only top correlations)
        for col in numeric_cols[:3]:  # Limit to 3 columns
            if col != target_col:
                correlation = df[target_col].corr(df[col])
                if abs(correlation) > 0.4:  # Only strong correlations
                    drivers.append({
                        "factor": col,
                        "correlation": correlation,
                        "impact": "High" if abs(correlation) > 0.7 else "Medium",
                        "direction": "Positive" if correlation > 0 else "Negative"
                    })
        
        return {
            "target_variable": target_col,
            "drivers": drivers[:3],  # Top 3 only
            "summary": f"Found {len(drivers)} key performance drivers for {target_col}"
        }
    
    def _identify_performance_drivers(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Identify key factors that drive business performance"""
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        
        if len(numeric_cols) < 2:
            return {"summary": "Insufficient numeric data for driver analysis", "drivers": []}
        
        # Find potential target variable (revenue, sales, profit, etc.)
        target_keywords = ['revenue', 'sales', 'profit', 'income', 'earnings', 'amount']
        target_col = None
        
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0]  # Use first numeric column as target
        
        drivers = []
        
        # Analyze correlation with target variable
        for col in numeric_cols:
            if col != target_col:
                correlation = df[target_col].corr(df[col])
                if abs(correlation) > 0.3:  # Significant correlation
                    impact = "High" if abs(correlation) > 0.7 else "Medium"
                    direction = "Positive" if correlation > 0 else "Negative"
                    
                    drivers.append({
                        "factor": col,
                        "correlation": correlation,
                        "impact": impact,
                        "direction": direction,
                        "recommendation": f"{'Increase' if correlation > 0 else 'Optimize'} {col} to improve {target_col}"
                    })
        
        # Analyze categorical drivers
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Avoid high cardinality categories
                group_performance = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                if len(group_performance) > 1:
                    top_performer = group_performance.index[0]
                    performance_gap = group_performance.iloc[0] - group_performance.iloc[-1]
                    
                    if performance_gap > 0:
                        drivers.append({
                            "factor": col,
                            "top_segment": top_performer,
                            "performance_gap": performance_gap,
                            "impact": "High" if performance_gap > df[target_col].std() else "Medium",
                            "recommendation": f"Focus resources on {top_performer} segment or replicate its success factors"
                        })
        
        # Sort drivers by impact
        numeric_drivers = sorted([d for d in drivers if 'correlation' in d], 
                               key=lambda x: abs(x['correlation']), reverse=True)
        categorical_drivers = sorted([d for d in drivers if 'performance_gap' in d], 
                                   key=lambda x: x['performance_gap'], reverse=True)
        
        summary = f"Identified {len(numeric_drivers)} numeric and {len(categorical_drivers)} categorical performance drivers for {target_col}"
        
        return {
            "target_variable": target_col,
            "numeric_drivers": numeric_drivers[:5],  # Top 5
            "categorical_drivers": categorical_drivers[:3],  # Top 3
            "summary": summary,
            "drivers": numeric_drivers + categorical_drivers
        }
    
    def _generate_strategic_recommendations_fast(self, df: pd.DataFrame, analysis: Dict, business_domain: str) -> List[Dict]:
        """Generate fast strategic recommendations with limited analysis"""
        recommendations = []
        numeric_cols = analysis['numeric_columns'][:3]  # Limit analysis
        
        # Quick revenue optimization
        revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'profit', 'amount'])]
        
        if revenue_cols:
            col = revenue_cols[0]
            q75 = df[col].quantile(0.75)
            below_target = len(df[df[col] < q75])
            
            recommendations.append({
                "category": "Revenue Optimization",
                "priority": "High",
                "recommendation": f"Improve {below_target} underperforming records",
                "action_items": [f"Focus on bottom quartile in {col}", "Implement best practices"]
            })
        
        # Quick data quality check
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > 5:
            recommendations.append({
                "category": "Data Quality",
                "priority": "Medium",
                "recommendation": f"Address {missing_pct:.1f}% missing data",
                "action_items": ["Improve data collection", "Set validation rules"]
            })
        
        return recommendations[:3]  # Return max 3 recommendations
    
    def _generate_strategic_recommendations(self, df: pd.DataFrame, analysis: Dict, business_domain: str) -> List[Dict]:
        """Generate strategic business recommendations based on data analysis"""
        recommendations = []
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        
        # Revenue optimization recommendations
        revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'profit', 'income', 'earnings', 'amount'])]
        
        if revenue_cols:
            col = revenue_cols[0]
            q75 = df[col].quantile(0.75)
            below_target = len(df[df[col] < q75])
            potential_increase = (q75 - df[df[col] < q75][col].mean()) * below_target
            
            recommendations.append({
                "category": "Revenue Optimization",
                "priority": "High",
                "recommendation": f"Improve {below_target} underperforming records to 75th percentile",
                "expected_impact": f"${potential_increase:,.0f} additional revenue potential",
                "action_items": [
                    f"Analyze top quartile performers in {col}",
                    "Identify success factors and best practices",
                    "Implement improvement programs for underperformers",
                    "Set performance targets at 75th percentile level"
                ]
            })
        
        # Market segmentation recommendations
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = revenue_cols[0] if revenue_cols else numeric_cols[0]
            
            segment_performance = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'sum']).sort_values('mean', ascending=False)
            top_segment = segment_performance.index[0]
            segment_opportunity = segment_performance['mean'].iloc[0] - segment_performance['mean'].mean()
            
            recommendations.append({
                "category": "Market Focus",
                "priority": "High",
                "recommendation": f"Prioritize {top_segment} segment for growth",
                "expected_impact": f"${segment_opportunity:,.0f} above-average performance per unit",
                "action_items": [
                    f"Increase investment in {top_segment} segment",
                    "Analyze what makes this segment successful",
                    "Develop segment-specific strategies",
                    "Consider expanding similar market segments"
                ]
            })
        
        # Operational efficiency recommendations
        if len(numeric_cols) >= 2:
            # Find high-variability metrics (improvement opportunities)
            high_var_cols = []
            for col in numeric_cols:
                cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
                if cv > 30:  # High coefficient of variation
                    high_var_cols.append((col, cv))
            
            if high_var_cols:
                col, cv = max(high_var_cols, key=lambda x: x[1])
                recommendations.append({
                    "category": "Operational Efficiency",
                    "priority": "Medium",
                    "recommendation": f"Reduce variability in {col}",
                    "expected_impact": f"Standardize performance (currently {cv:.1f}% variation)",
                    "action_items": [
                        f"Investigate root causes of {col} variability",
                        "Implement standardized processes",
                        "Establish quality control measures",
                        "Monitor performance consistency"
                    ]
                })
        
        # Data quality recommendations
        missing_data_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_data_pct > 5:
            recommendations.append({
                "category": "Data Quality",
                "priority": "Medium",
                "recommendation": "Improve data collection and completeness",
                "expected_impact": f"Reduce {missing_data_pct:.1f}% missing data for better insights",
                "action_items": [
                    "Audit data collection processes",
                    "Implement data validation rules",
                    "Train staff on data entry standards",
                    "Set up automated data quality monitoring"
                ]
            })
        
        # Growth opportunity recommendations
        if revenue_cols and categorical_cols:
            # Find underperforming segments with potential
            growth_opportunities = []
            for cat_col in categorical_cols[:2]:  # Check top 2 categorical columns
                segment_data = df.groupby(cat_col)[revenue_cols[0]].agg(['mean', 'count']).reset_index()
                segment_data['potential'] = segment_data['mean'] * segment_data['count']
                
                # Find segments with high count but below-average performance
                avg_performance = df[revenue_cols[0]].mean()
                opportunities = segment_data[
                    (segment_data['count'] >= segment_data['count'].quantile(0.6)) &
                    (segment_data['mean'] < avg_performance)
                ]
                
                if not opportunities.empty:
                    top_opp = opportunities.loc[opportunities['potential'].idxmax()]
                    growth_opportunities.append({
                        "segment_type": cat_col,
                        "segment": top_opp[cat_col],
                        "current_performance": top_opp['mean'],
                        "volume": top_opp['count'],
                        "improvement_potential": avg_performance - top_opp['mean']
                    })
            
            if growth_opportunities:
                best_opp = max(growth_opportunities, key=lambda x: x['improvement_potential'] * x['volume'])
                recommendations.append({
                    "category": "Growth Opportunity",
                    "priority": "High",
                    "recommendation": f"Focus improvement efforts on {best_opp['segment']} segment",
                    "expected_impact": f"${(best_opp['improvement_potential'] * best_opp['volume']):,.0f} potential value increase",
                    "action_items": [
                        f"Develop targeted improvement plan for {best_opp['segment']}",
                        "Benchmark against top-performing segments",
                        "Allocate additional resources to this opportunity",
                        "Set specific performance improvement targets"
                    ]
                })
        
        return recommendations
    
    def _perform_predictive_modeling_fast(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Fast predictive modeling with simplified approach"""
        numeric_cols = analysis['numeric_columns'][:3]  # Limit to 3 columns
        
        if len(numeric_cols) < 2:
            return {"summary": "Insufficient data for modeling", "predictions": []}
        
        # Quick target detection
        target_col = numeric_cols[0]
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'profit']):
                target_col = col
                break
        
        try:
            feature_cols = [col for col in numeric_cols if col != target_col][:2]  # Max 2 features
            
            if not feature_cols:
                return {"summary": "No features for prediction", "predictions": []}
            
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col].fillna(df[target_col].mean())
            
            if len(df) > 20:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Fast model with limited complexity
                model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                # Top feature only
                feature_importance = list(zip(feature_cols, model.feature_importances_))
                top_feature = max(feature_importance, key=lambda x: x[1])
                
                return {
                    "model_performance": {"rmse": np.sqrt(mse)},
                    "top_feature": top_feature[0],
                    "summary": f"Fast model trained for {target_col}",
                    "predictions": [f"Key predictor: {top_feature[0]}"]
                }
            else:
                return {"summary": "Dataset too small for modeling", "predictions": []}
                
        except Exception as e:
            return {"summary": f"Modeling failed: {str(e)}", "predictions": []}
    
    def _perform_predictive_modeling(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Perform predictive modeling to forecast future performance"""
        numeric_cols = analysis['numeric_columns']
        
        if len(numeric_cols) < 2:
            return {"summary": "Insufficient data for predictive modeling", "predictions": []}
        
        # Find target variable
        target_keywords = ['revenue', 'sales', 'profit', 'income', 'earnings', 'amount']
        target_col = None
        
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0]
        
        try:
            # Prepare features (exclude target)
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            if len(feature_cols) == 0:
                return {"summary": "No features available for prediction", "predictions": []}
            
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col].fillna(df[target_col].mean())
            
            # Simple train-test split
            if len(df) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Random Forest model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                # Feature importance
                feature_importance = list(zip(feature_cols, model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Generate predictions for different scenarios
                scenarios = self._generate_scenarios(X, model, feature_cols, target_col)
                
                return {
                    "model_performance": {
                        "mse": mse,
                        "rmse": np.sqrt(mse),
                        "accuracy": "Good" if mse < y.var() else "Fair"
                    },
                    "feature_importance": feature_importance[:5],
                    "scenarios": scenarios,
                    "summary": f"Predictive model trained with RMSE of {np.sqrt(mse):.2f} for {target_col}",
                    "predictions": [
                        f"Most important predictor: {feature_importance[0][0]}",
                        f"Model explains variance in {target_col}",
                        f"Can forecast {target_col} based on {len(feature_cols)} factors"
                    ]
                }
            else:
                return {"summary": "Dataset too small for reliable predictive modeling", "predictions": []}
                
        except Exception as e:
            return {"summary": f"Predictive modeling failed: {str(e)}", "predictions": []}
    
    def _generate_scenarios(self, X: pd.DataFrame, model, feature_cols: List[str], target_col: str) -> Dict[str, float]:
        """Generate what-if scenarios using the trained model"""
        scenarios = {}
        
        # Base scenario (current average)
        base_features = X.mean().values.reshape(1, -1)
        base_prediction = model.predict(base_features)[0]
        scenarios['current_average'] = base_prediction
        
        # Optimistic scenario (top quartile values)
        optimistic_features = X.quantile(0.75).values.reshape(1, -1)
        optimistic_prediction = model.predict(optimistic_features)[0]
        scenarios['optimistic'] = optimistic_prediction
        
        # Conservative scenario (bottom quartile values)
        conservative_features = X.quantile(0.25).values.reshape(1, -1)
        conservative_prediction = model.predict(conservative_features)[0]
        scenarios['conservative'] = conservative_prediction
        
        # Best case scenario (maximum values)
        best_case_features = X.max().values.reshape(1, -1)
        best_case_prediction = model.predict(best_case_features)[0]
        scenarios['best_case'] = best_case_prediction
        
        return scenarios
    
    def _identify_optimization_opportunities(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Identify specific optimization opportunities in the data"""
        opportunities = {}
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        
        # Revenue optimization
        revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'profit', 'income', 'earnings', 'amount'])]
        
        if revenue_cols:
            col = revenue_cols[0]
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            
            underperformers = df[df[col] <= q25]
            improvement_potential = (q75 - q25) * len(underperformers)
            
            opportunities['revenue_optimization'] = {
                "type": "Revenue Growth",
                "opportunity": f"Improve bottom quartile performance in {col}",
                "potential_value": improvement_potential,
                "affected_records": len(underperformers),
                "recommendation": f"Focus on {len(underperformers)} underperforming records",
                "target_improvement": q75 - q25
            }
        
        # Efficiency optimization
        if len(numeric_cols) >= 2:
            efficiency_opportunities = []
            for col in numeric_cols:
                cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
                if cv > 25:  # High variability indicates optimization opportunity
                    efficiency_opportunities.append({
                        "metric": col,
                        "variability": cv,
                        "standardization_opportunity": df[col].std()
                    })
            
            if efficiency_opportunities:
                top_opportunity = max(efficiency_opportunities, key=lambda x: x['variability'])
                opportunities['efficiency_optimization'] = {
                    "type": "Process Standardization",
                    "opportunity": f"Reduce variability in {top_opportunity['metric']}",
                    "current_variability": f"{top_opportunity['variability']:.1f}%",
                    "recommendation": f"Standardize processes affecting {top_opportunity['metric']}",
                    "potential_improvement": "20-30% reduction in variability"
                }
        
        # Market focus optimization
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = revenue_cols[0] if revenue_cols else numeric_cols[0]
            
            segment_performance = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'sum']).reset_index()
            segment_performance['efficiency'] = segment_performance['sum'] / segment_performance['count']
            
            top_segment = segment_performance.loc[segment_performance['efficiency'].idxmax()]
            total_opportunity = (top_segment['efficiency'] - segment_performance['efficiency'].mean()) * segment_performance['count'].sum()
            
            opportunities['market_focus'] = {
                "type": "Market Optimization",
                "opportunity": f"Focus on high-performing {cat_col} segments",
                "top_segment": top_segment[cat_col],
                "efficiency_advantage": top_segment['efficiency'] - segment_performance['efficiency'].mean(),
                "total_opportunity": total_opportunity,
                "recommendation": f"Prioritize {top_segment[cat_col]} segment strategy"
            }
        
        return opportunities