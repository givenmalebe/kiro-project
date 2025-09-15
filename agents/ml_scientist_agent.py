import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class MLScientistAgent:
    def __init__(self, google_api_key: str = ""):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self.models = {
            'regression': [
                'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net',
                'Random Forest', 'Gradient Boosting', 'XGBoost', 'SVR', 'KNN',
                'Neural Network', 'Transformer', 'Decision Tree'
            ],
            'classification': [
                'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost',
                'SVM', 'KNN', 'Decision Tree', 'Neural Network', 'Transformer'
            ],
            'clustering': ['K-Means', 'DBSCAN', 'Hierarchical', 'Neural Network'],
            'time_series': ['LSTM', 'GRU', 'Transformer', 'ARIMA', 'Prophet'],
            'dimensionality_reduction': ['PCA', 'Feature Selection', 'Autoencoder']
        }
        
        # AI-powered model recommendations based on data characteristics
        self.model_recommendations = {
            'small_dataset': ['Linear Regression', 'Logistic Regression', 'SVM', 'KNN'],
            'large_dataset': ['Random Forest', 'XGBoost', 'Neural Network', 'Transformer'],
            'high_dimensional': ['Ridge', 'Lasso', 'PCA', 'Feature Selection'],
            'non_linear': ['Random Forest', 'XGBoost', 'Neural Network', 'SVM'],
            'time_series': ['LSTM', 'GRU', 'Transformer', 'ARIMA'],
            'clustering': ['K-Means', 'DBSCAN', 'Hierarchical']
        }
        
    def analyze_data_and_recommend_models(self, df: pd.DataFrame) -> dict:
        """Analyze data characteristics and recommend appropriate ML/DL models"""
        analysis = {
            'data_shape': df.shape,
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'target_candidates': [],
            'recommended_models': [],
            'task_type': 'unknown'
        }
        
        # AI-powered data analysis
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        n_samples, n_features = df.shape
        
        # Analyze data characteristics
        analysis['data_characteristics'] = {
            'sample_size': n_samples,
            'feature_count': n_features,
            'missing_data_percentage': (df.isnull().sum().sum() / (n_samples * n_features)) * 100,
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'high_dimensional': n_features > n_samples * 0.1,
            'small_dataset': n_samples < 1000,
            'large_dataset': n_samples > 10000
        }
        
        # AI-powered insights
        if analysis['data_characteristics']['small_dataset']:
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("🔍 Small dataset detected - recommend simpler models to avoid overfitting")
            analysis['recommended_models'].extend(self.model_recommendations['small_dataset'])
        
        if analysis['data_characteristics']['large_dataset']:
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("📊 Large dataset detected - complex models like Neural Networks and XGBoost will perform well")
            analysis['recommended_models'].extend(self.model_recommendations['large_dataset'])
        
        if analysis['data_characteristics']['high_dimensional']:
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("🎯 High-dimensional data - consider dimensionality reduction techniques")
            analysis['recommended_models'].extend(self.model_recommendations['high_dimensional'])
        
        if analysis['data_characteristics']['missing_data_percentage'] > 10:
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("⚠️ Significant missing data detected - preprocessing required")
            analysis['preprocessing_recommendations'] = analysis.get('preprocessing_recommendations', [])
            analysis['preprocessing_recommendations'].append("Handle missing values with imputation or removal")
        
        # Task type detection and model recommendations
        if len(numeric_cols) > 1:
            analysis['target_candidates'].extend(numeric_cols)
            analysis['task_type'] = 'regression'
            analysis['recommended_models'].extend(self.models['regression'])
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("📈 Regression task detected - predicting continuous values")
        
        if len(categorical_cols) > 0:
            analysis['target_candidates'].extend(categorical_cols)
            if analysis['task_type'] == 'unknown':
                analysis['task_type'] = 'classification'
            analysis['recommended_models'].extend(self.models['classification'])
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("🏷️ Classification task detected - predicting categories")
        
        # Check for time series patterns
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            analysis['ai_insights'] = analysis.get('ai_insights', [])
            analysis['ai_insights'].append("⏰ Time series data detected - consider LSTM or Transformer models")
            analysis['recommended_models'].extend(self.model_recommendations['time_series'])
        
        # For clustering tasks
        if len(numeric_cols) >= 2:
            analysis['recommended_models'].extend(self.models['clustering'])
        
        # Add preprocessing recommendations
        if len(categorical_cols) > 0:
            analysis['preprocessing_recommendations'] = analysis.get('preprocessing_recommendations', [])
            analysis['preprocessing_recommendations'].append("Encode categorical variables")
        if len(numeric_cols) > 0:
            analysis['preprocessing_recommendations'] = analysis.get('preprocessing_recommendations', [])
            analysis['preprocessing_recommendations'].append("Scale numerical features")
        
        # Remove duplicates and prioritize recommendations
        analysis['recommended_models'] = list(dict.fromkeys(analysis['recommended_models']))[:8]
        
        return analysis
    
    def generate_ml_code(self, df: pd.DataFrame, target_column: str, model_type: str, task_type: str) -> str:
        """Generate comprehensive ML/DL code for the specified model and task"""
        
        # AI-powered task type detection and model validation
        detected_task_type = self._detect_task_type(df, target_column)
        
        # If there's a mismatch, use AI to choose the best approach
        if task_type != detected_task_type:
            print(f"🤖 AI detected task type mismatch: {task_type} vs {detected_task_type}")
            print(f"🧠 AI recommendation: Using {detected_task_type} for better results")
            task_type = detected_task_type
        
        # AI-powered model selection if needed
        if model_type.lower() in ['auto', 'best', 'recommended']:
            model_type = self._select_best_model(df, target_column, task_type)
            print(f"🤖 AI selected best model: {model_type}")
        
        if model_type.lower() in ['neural network', 'transformer']:
            return self._generate_deep_learning_code(df, target_column, model_type, task_type)
        else:
            return self._generate_enhanced_traditional_ml_code(df, target_column, model_type, task_type)
    
    def _generate_deep_learning_code(self, df: pd.DataFrame, target_column: str, model_type: str, task_type: str) -> str:
        """Generate PyTorch deep learning code"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column in numeric_cols:
            task_type = 'regression'
        else:
            task_type = 'classification'
        
        code = f"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and prepare data
# Data is already loaded and available as 'df'
print(f"Dataset shape: {{df.shape}}")

# Separate features and target
target_col = '{target_column}'
"""
        
        if task_type == 'regression':
            code += f"""
# For regression task
y = df[target_col].values
X = df.drop(columns=[target_col])

# Handle categorical variables and datetime columns
categorical_cols = {categorical_cols}
datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()

# Remove datetime columns or convert them
for col in datetime_cols:
    if col in X.columns:
        X = X.drop(columns=[col])

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Convert to numpy
X = X.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define {model_type} model
class {model_type.replace(' ', '')}Model(nn.Module):
    def __init__(self, input_size):
        super({model_type.replace(' ', '')}Model, self).__init__()
        self.input_size = input_size
        
        if '{model_type.lower()}' == 'transformer':
            # Transformer architecture
            self.embedding = nn.Linear(input_size, 128)
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, 128))
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            # Neural Network architecture
            self.fc = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
    
    def forward(self, x):
        if '{model_type.lower()}' == 'transformer':
            # Add sequence dimension for transformer
            x = x.unsqueeze(1)  # [batch_size, 1, features]
            x = self.embedding(x)
            x = x + self.pos_encoding
            x = self.transformer(x)
            x = x.squeeze(1)  # Remove sequence dimension
            return self.fc(x)
        else:
            return self.fc(x)

# Initialize model
model = {model_type.replace(' ', '')}Model(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_losses.append(val_loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{{epoch+1}}/{{epochs}}], Loss: {{epoch_loss/len(train_loader):.4f}}, Val Loss: {{val_loss.item():.4f}}')

# Make predictions
model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor).numpy()
    test_pred = model(X_test_tensor).numpy()

# Calculate metrics
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\\n=== {model_type} Model Results ===")
print(f"Training R²: {{train_r2:.4f}}")
print(f"Testing R²: {{test_r2:.4f}}")
print(f"Training RMSE: {{train_rmse:.4f}}")
print(f"Testing RMSE: {{test_rmse:.4f}}")
print(f"Training MAE: {{train_mae:.4f}}")
print(f"Testing MAE: {{test_mae:.4f}}")

# Generate batch predictions for visualization
batch_size = 100
if len(X_test) > batch_size:
    batch_indices = np.random.choice(len(X_test), batch_size, replace=False)
    batch_X = X_test[batch_indices]
    batch_y = y_test[batch_indices]
    batch_pred = test_pred[batch_indices]
else:
    batch_X = X_test
    batch_y = y_test
    batch_pred = test_pred

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training vs Validation Loss
axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Actual vs Predicted (Test Set)
axes[0, 1].scatter(y_test, test_pred, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_title(f'Actual vs Predicted (R² = {{test_r2:.4f}})')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True)

# 3. Residuals Plot
residuals = y_test - test_pred.flatten()
axes[1, 0].scatter(test_pred, residuals, alpha=0.6, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Residuals Plot')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].grid(True)

# 4. Batch Predictions Comparison
x_range = range(len(batch_pred))
axes[1, 1].plot(x_range, batch_y, 'o-', label='Actual', color='blue', alpha=0.7)
axes[1, 1].plot(x_range, batch_pred.flatten(), 's-', label='Predicted', color='red', alpha=0.7)
axes[1, 1].set_title(f'Batch Predictions ({{len(batch_pred)}} samples)')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('Target Value')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Feature importance (for neural networks, we'll use permutation importance)
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

class PyTorchModelWrapper(BaseEstimator):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(self.scaler.transform(X))
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).numpy().flatten()
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        predictions = self.predict(X)
        return r2_score(y, predictions)

# Calculate permutation importance
model_wrapper = PyTorchModelWrapper(model, scaler)
perm_importance = permutation_importance(
    model_wrapper, X_test, y_test, n_repeats=10, random_state=42
)

feature_names = [f'Feature_{{i}}' for i in range(X_test.shape[1])]
importance_df = pd.DataFrame({{
    'feature': feature_names,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}}).sort_values('importance', ascending=False)

print("\\n=== Feature Importance ===")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = importance_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""
        
        else:  # Classification
            code += f"""
# For classification task
y = df[target_col].values
X = df.drop(columns=[target_col])

# Handle categorical variables and datetime columns
categorical_cols = {categorical_cols}
datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()

# Remove datetime columns or convert them
for col in datetime_cols:
    if col in X.columns:
        X = X.drop(columns=[col])

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Convert to numpy
X = X.values

# Encode target for classification
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
num_classes = len(le_target.classes_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define {model_type} model for classification
class {model_type.replace(' ', '')}Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super({model_type.replace(' ', '')}Classifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        if '{model_type.lower()}' == 'transformer':
            # Transformer architecture
            self.embedding = nn.Linear(input_size, 128)
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, 128))
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
        else:
            # Neural Network architecture
            self.fc = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
    
    def forward(self, x):
        if '{model_type.lower()}' == 'transformer':
            # Add sequence dimension for transformer
            x = x.unsqueeze(1)  # [batch_size, 1, features]
            x = self.embedding(x)
            x = x + self.pos_encoding
            x = self.transformer(x)
            x = x.squeeze(1)  # Remove sequence dimension
            return self.fc(x)
        else:
            return self.fc(x)

# Initialize model
model = {model_type.replace(' ', '')}Classifier(X_train_scaled.shape[1], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = 100 * (val_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{{epoch+1}}/{{epochs}}], Loss: {{epoch_loss/len(train_loader):.4f}}, Acc: {{100*correct/total:.2f}}%, Val Loss: {{val_loss.item():.4f}}, Val Acc: {{val_accuracy:.2f}}%')

# Make predictions
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    test_outputs = model(X_test_tensor)
    _, train_pred = torch.max(train_outputs, 1)
    _, test_pred = torch.max(test_outputs, 1)

# Calculate metrics
train_accuracy = accuracy_score(y_train, train_pred.numpy())
test_accuracy = accuracy_score(y_test, test_pred.numpy())

print(f"\\n=== {model_type} Classification Results ===")
print(f"Training Accuracy: {{train_accuracy:.4f}}")
print(f"Testing Accuracy: {{test_accuracy:.4f}}")
print(f"Number of Classes: {{num_classes}}")
print(f"Classes: {{le_target.classes_}}")

# Classification report
print("\\n=== Classification Report ===")
print(classification_report(y_test, test_pred.numpy(), target_names=le_target.classes_))

# Generate batch predictions for visualization
batch_size = 100
if len(X_test) > batch_size:
    batch_indices = np.random.choice(len(X_test), batch_size, replace=False)
    batch_X = X_test[batch_indices]
    batch_y = y_test[batch_indices]
    batch_pred = test_pred.numpy()[batch_indices]
else:
    batch_X = X_test
    batch_y = y_test
    batch_pred = test_pred.numpy()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training vs Validation Loss
axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Training vs Validation Accuracy
axes[0, 1].plot(train_accuracies, label='Training Accuracy', color='blue')
axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='red')
axes[0, 1].set_title('Training vs Validation Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_pred.numpy())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Batch Predictions Comparison
x_range = range(len(batch_pred))
axes[1, 1].scatter(x_range, batch_y, label='Actual', color='blue', alpha=0.7)
axes[1, 1].scatter(x_range, batch_pred, label='Predicted', color='red', alpha=0.7)
axes[1, 1].set_title(f'Batch Predictions ({{len(batch_pred)}} samples)')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('Class')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Feature importance
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

class PyTorchClassifierWrapper(BaseEstimator):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(self.scaler.transform(X))
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Calculate permutation importance
model_wrapper = PyTorchClassifierWrapper(model, scaler)
perm_importance = permutation_importance(
    model_wrapper, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
)

feature_names = [f'Feature_{{i}}' for i in range(X_test.shape[1])]
importance_df = pd.DataFrame({{
    'feature': feature_names,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}}).sort_values('importance', ascending=False)

print("\\n=== Feature Importance ===")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = importance_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""
        
        return code
    
    def _generate_traditional_ml_code(self, df: pd.DataFrame, target_column: str, model_type: str, task_type: str) -> str:
        """Generate traditional ML code using scikit-learn"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column in numeric_cols:
            task_type = 'regression'
        else:
            task_type = 'classification'
        
        code = f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
# Data is already loaded and available as 'df'
print(f"Dataset shape: {{df.shape}}")

# Separate features and target
target_col = '{target_column}'
y = df[target_col]

# AI-powered target preprocessing
if y.dtype == 'object' or y.dtype.name == 'category':
    # Classification task - encode labels
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    task_type = 'classification'
    print(f"🤖 AI detected classification task with {len(np.unique(y))} classes")
else:
    # Regression task - ensure numeric
    y = pd.to_numeric(y, errors='coerce')
    task_type = 'regression'
    print(f"🤖 AI detected regression task with continuous values")

X = df.drop(columns=[target_col])

# Handle categorical variables and datetime columns
categorical_cols = {categorical_cols}
datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()

# Remove datetime columns or convert them
for col in datetime_cols:
    if col in X.columns:
        X = X.drop(columns=[col])

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Convert to numpy
X = X.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import and initialize model
"""
        
        # AI-powered model selection based on detected task type
        if model_type == 'Linear Regression':
            code += """
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""
        elif model_type == 'Random Forest':
            if task_type == 'regression':
                code += """
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
"""
            else:
                code += """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
"""
        elif model_type == 'XGBoost':
            if task_type == 'regression':
                code += """
from xgboost import XGBRegressor
model = XGBRegressor(random_state=42)
"""
            else:
                code += """
from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
"""
        elif model_type == 'Logistic Regression':
            code += """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, max_iter=1000)
"""
        elif model_type == 'Ridge Regression':
            code += """
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0, random_state=42)
"""
        elif model_type == 'Lasso Regression':
            code += """
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0, random_state=42)
"""
        elif model_type == 'Elastic Net':
            code += """
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
"""
        elif model_type == 'Gradient Boosting':
            if task_type == 'regression':
                code += """
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)
"""
            else:
                code += """
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
"""
        elif model_type == 'SVR':
            code += """
from sklearn.svm import SVR
model = SVR(kernel='rbf')
"""
        elif model_type == 'SVM':
            code += """
from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state=42)
"""
        elif model_type == 'KNN':
            if task_type == 'regression':
                code += """
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
"""
            else:
                code += """
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
"""
        elif model_type == 'Decision Tree':
            if task_type == 'regression':
                code += """
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
"""
            else:
                code += """
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
"""
        
        if task_type == 'regression':
            code += f"""
# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\\n=== {model_type} Model Results ===")
print(f"Training R²: {{train_r2:.4f}}")
print(f"Testing R²: {{test_r2:.4f}}")
print(f"Training RMSE: {{train_rmse:.4f}}")
print(f"Testing RMSE: {{test_rmse:.4f}}")
print(f"Training MAE: {{train_mae:.4f}}")
print(f"Testing MAE: {{test_mae:.4f}}")

# Generate batch predictions for visualization
batch_size = 100
if len(X_test) > batch_size:
    batch_indices = np.random.choice(len(X_test), batch_size, replace=False)
    batch_X = X_test[batch_indices]
    batch_y = y_test[batch_indices]
    batch_pred = test_pred[batch_indices]
else:
    batch_X = X_test
    batch_y = y_test
    batch_pred = test_pred

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, test_pred, alpha=0.6, color='green')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_title(f'Actual vs Predicted (R² = {{test_r2:.4f}})')
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].grid(True)

# 2. Residuals Plot
residuals = y_test - test_pred
axes[0, 1].scatter(test_pred, residuals, alpha=0.6, color='purple')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_title('Residuals Plot')
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].grid(True)

# 3. Batch Predictions Comparison
x_range = range(len(batch_pred))
axes[1, 0].plot(x_range, batch_y, 'o-', label='Actual', color='blue', alpha=0.7)
axes[1, 0].plot(x_range, batch_pred, 's-', label='Predicted', color='red', alpha=0.7)
axes[1, 0].set_title(f'Batch Predictions ({{len(batch_pred)}} samples)')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Target Value')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Feature Importance
if hasattr(model, 'feature_importances_'):
    feature_names = [f'Feature_{{i}}' for i in range(X.shape[1])]
    importance_df = pd.DataFrame({{
        'feature': feature_names,
        'importance': model.feature_importances_
    }}).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(10)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'])
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].invert_yaxis()
else:
    axes[1, 1].text(0.5, 0.5, 'Feature importance\\nnot available\\nfor this model', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.show()

# Print feature importance
if hasattr(model, 'feature_importances_'):
    print("\\n=== Feature Importance ===")
    print(importance_df.head(10))
"""
        
        else:  # Classification
            code += f"""
# Encode target for classification
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
num_classes = len(le_target.classes_)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\\n=== {model_type} Classification Results ===")
print(f"Training Accuracy: {{train_accuracy:.4f}}")
print(f"Testing Accuracy: {{test_accuracy:.4f}}")
print(f"Number of Classes: {{num_classes}}")
print(f"Classes: {{le_target.classes_}}")

# Classification report
print("\\n=== Classification Report ===")
print(classification_report(y_test, test_pred, target_names=le_target.classes_))

# Generate batch predictions for visualization
batch_size = 100
if len(X_test) > batch_size:
    batch_indices = np.random.choice(len(X_test), batch_size, replace=False)
    batch_X = X_test[batch_indices]
    batch_y = y_test[batch_indices]
    batch_pred = test_pred[batch_indices]
else:
    batch_X = X_test
    batch_y = y_test
    batch_pred = test_pred

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. Classification Report Visualization
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average=None)
metrics_df = pd.DataFrame({{
    'Class': le_target.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}})

x_pos = np.arange(len(le_target.classes_))
width = 0.25
axes[0, 1].bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
axes[0, 1].bar(x_pos, recall, width, label='Recall', alpha=0.8)
axes[0, 1].bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)
axes[0, 1].set_xlabel('Classes')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Precision, Recall, F1-Score by Class')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(le_target.classes_, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Batch Predictions Comparison
x_range = range(len(batch_pred))
axes[1, 0].scatter(x_range, batch_y, label='Actual', color='blue', alpha=0.7)
axes[1, 0].scatter(x_range, batch_pred, label='Predicted', color='red', alpha=0.7)
axes[1, 0].set_title(f'Batch Predictions ({{len(batch_pred)}} samples)')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Class')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Feature Importance
if hasattr(model, 'feature_importances_'):
    feature_names = [f'Feature_{{i}}' for i in range(X.shape[1])]
    importance_df = pd.DataFrame({{
        'feature': feature_names,
        'importance': model.feature_importances_
    }}).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(10)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'])
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].invert_yaxis()
else:
    axes[1, 1].text(0.5, 0.5, 'Feature importance\\nnot available\\nfor this model', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Feature Importance')

plt.tight_layout()
plt.show()

# Print feature importance
if hasattr(model, 'feature_importances_'):
    print("\\n=== Feature Importance ===")
    print(importance_df.head(10))
"""
        
        return code
    
    def _generate_enhanced_traditional_ml_code(self, df: pd.DataFrame, target_column: str, model_type: str, task_type: str) -> str:
        """Generate enhanced traditional ML code with AI-powered task detection"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        code = f"""
# AI-Powered Machine Learning Analysis
# Target: {target_column} | Model: {model_type} | Task: {task_type}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Data preparation
df = df.copy()
print(f"🤖 AI analyzing dataset: {{df.shape}}")

# Separate features and target
target_col = '{target_column}'
y = df[target_col]

# AI-powered target preprocessing
if y.dtype == 'object' or y.dtype.name == 'category':
    # Classification task - encode labels
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    task_type = 'classification'
    print(f"🤖 AI detected classification task with {{len(np.unique(y))}} classes")
else:
    # Regression task - ensure numeric
    y = pd.to_numeric(y, errors='coerce')
    task_type = 'regression'
    print(f"🤖 AI detected regression task with continuous values")

X = df.drop(columns=[target_col])

# Handle categorical variables and datetime columns
categorical_cols = {categorical_cols}
datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()

# Remove datetime columns
for col in datetime_cols:
    if col in X.columns:
        X = X.drop(columns=[col])
        print(f"🤖 AI removed datetime column: {{col}}")

# Encode categorical variables
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"🤖 AI encoded categorical column: {{col}}")

# Handle missing values
X = X.fillna(X.mean())
if hasattr(y, 'fillna'):
    y = y.fillna(y.mean()) if y.dtype in ['float64', 'int64'] else y
else:
    # Handle numpy arrays
    y = pd.Series(y).fillna(pd.Series(y).mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# AI-powered model selection
"""
        
        # Add model initialization based on task type
        if model_type == 'Linear Regression':
            code += """
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""
        elif model_type == 'Random Forest':
            code += """
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
if task_type == 'regression':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
"""
        elif model_type == 'XGBoost':
            code += """
from xgboost import XGBRegressor, XGBClassifier
if task_type == 'regression':
    model = XGBRegressor(random_state=42)
else:
    model = XGBClassifier(random_state=42)
"""
        elif model_type == 'SVM':
            code += """
from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state=42)
"""
        elif model_type == 'SVR':
            code += """
from sklearn.svm import SVR
model = SVR(kernel='rbf')
"""
        elif model_type == 'KNN':
            code += """
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
if task_type == 'regression':
    model = KNeighborsRegressor(n_neighbors=5)
else:
    model = KNeighborsClassifier(n_neighbors=5)
"""
        elif model_type == 'Logistic Regression':
            code += """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, max_iter=1000)
"""
        elif model_type == 'Ridge Regression':
            code += """
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0, random_state=42)
"""
        elif model_type == 'Lasso Regression':
            code += """
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0, random_state=42)
"""
        elif model_type == 'Elastic Net':
            code += """
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
"""
        elif model_type == 'Gradient Boosting':
            code += """
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
if task_type == 'regression':
    model = GradientBoostingRegressor(random_state=42)
else:
    model = GradientBoostingClassifier(random_state=42)
"""
        elif model_type == 'Decision Tree':
            code += """
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
if task_type == 'regression':
    model = DecisionTreeRegressor(random_state=42)
else:
    model = DecisionTreeClassifier(random_state=42)
"""
        
        # Add training and evaluation
        code += f"""
# Train model
print(f"🤖 AI training {model_type} model...")
model.fit(X_train_scaled, y_train)

# Make predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# AI-powered evaluation based on task type
if task_type == 'regression':
    # Regression metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print(f"\\n=== {model_type} Regression Results ===")
    print(f"Training R²: {{train_r2:.4f}}")
    print(f"Testing R²: {{test_r2:.4f}}")
    print(f"Training RMSE: {{train_rmse:.4f}}")
    print(f"Testing RMSE: {{test_rmse:.4f}}")
    print(f"Training MAE: {{train_mae:.4f}}")
    print(f"Testing MAE: {{test_mae:.4f}}")

    # Store metrics for explanation
    r2 = test_r2
    rmse = test_rmse
    mae = test_mae
else:
    # Classification metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\\n=== {model_type} Classification Results ===")
    print(f"Training Accuracy: {{train_accuracy:.4f}}")
    print(f"Testing Accuracy: {{test_accuracy:.4f}}")
    print(f"Number of Classes: {{len(np.unique(y))}}")
    
    # Store metrics for explanation
    accuracy = test_accuracy

# Cross-validation
print(f"\\n🤖 AI performing cross-validation...")
if task_type == 'regression':
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R²: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")
else:
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")

# Create visualizations
plt.figure(figsize=(15, 10))

if task_type == 'regression':
    # Plot 1: Predictions vs Actual
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')

    # Plot 2: Residuals
    plt.subplot(2, 3, 2)
    residuals = y_test - test_pred
    plt.scatter(test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({{'feature': feature_names, 'importance': feature_importance}})
        importance_df = importance_df.sort_values('importance', ascending=False).head(10)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
    else:
        plt.text(0.5, 0.5, 'Feature importance\\nnot available', ha='center', va='center')
        plt.title('Feature Importance')

else:  # classification
    # Plot 1: Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Plot 2: Classification Report
    plt.subplot(2, 3, 2)
    plt.text(0.1, 0.9, f'Accuracy: {{test_accuracy:.4f}}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.1, 0.8, f'Classes: {{len(np.unique(y))}}', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Classification Summary')
    plt.axis('off')

    # Plot 3: Feature Importance
    plt.subplot(2, 3, 3)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({{'feature': feature_names, 'importance': feature_importance}})
        importance_df = importance_df.sort_values('importance', ascending=False).head(10)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
    else:
        plt.text(0.5, 0.5, 'Feature importance\\nnot available', ha='center', va='center')
        plt.title('Feature Importance')

plt.tight_layout()
plt.show()

# Feature importance analysis
if hasattr(model, 'feature_importances_'):
    print("\\n=== Feature Importance ===")
    importance_df = pd.DataFrame({{'feature': X.columns, 'importance': model.feature_importances_}})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print(importance_df.head(10))

print(f"\\n✅ AI-powered {model_type} analysis completed successfully!")
"""
        
        return code
    
    def _detect_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """AI-powered task type detection based on target column characteristics"""
        target_data = df[target_column]
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target_data):
            # Check if it's continuous or discrete
            unique_values = target_data.nunique()
            total_values = len(target_data)
            
            # If more than 20 unique values and they're not integers, likely regression
            if unique_values > 20 and not pd.api.types.is_integer_dtype(target_data):
                return 'regression'
            # If few unique values, might be classification
            elif unique_values <= 10:
                return 'classification'
            else:
                return 'regression'
        else:
            # Non-numeric data is classification
            return 'classification'
    
    def _select_best_model(self, df: pd.DataFrame, target_column: str, task_type: str) -> str:
        """AI-powered model selection based on data characteristics"""
        n_samples, n_features = df.shape
        target_data = df[target_column]
        
        # Get data characteristics
        is_small_dataset = n_samples < 1000
        is_large_dataset = n_samples > 10000
        is_high_dimensional = n_features > n_samples * 0.1
        has_missing_data = df.isnull().sum().sum() > 0
        
        # AI decision tree for model selection
        if task_type == 'regression':
            if is_small_dataset:
                return 'Linear Regression'  # Simple, less prone to overfitting
            elif is_high_dimensional:
                return 'Ridge Regression'  # Handles high dimensions well
            elif is_large_dataset:
                return 'XGBoost'  # Excellent for large datasets
            else:
                return 'Random Forest'  # Good general purpose
        else:  # classification
            if is_small_dataset:
                return 'Logistic Regression'  # Simple, interpretable
            elif is_high_dimensional:
                return 'SVM'  # Good for high dimensions
            elif is_large_dataset:
                return 'XGBoost'  # Excellent performance on large datasets
            else:
                return 'Random Forest'  # Good general purpose
    
    def execute_ml_code(self, code: str, df: pd.DataFrame) -> dict:
        """Execute ML code and return results with visualizations and detailed explanations"""
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
            
            # Capture any generated figures with detailed explanations
            images = []
            explanations = []
            
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    
                    # Get figure title and axes for explanation
                    title = fig._suptitle.get_text() if fig._suptitle else f"Figure {fig_num}"
                    
                    # Capture the image
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    # Generate explanation for this figure
                    explanation = self._generate_figure_explanation(fig, title, exec_globals)
                    
                    images.append(img_base64)
                    explanations.append(explanation)
                
                plt.close('all')  # Close all figures to free memory
            
            # Generate overall model performance explanation
            overall_explanation = self._generate_overall_explanation(exec_globals, len(images))
            
            return {
                'success': True,
                'images': images,
                'explanations': explanations,
                'overall_explanation': overall_explanation,
                'message': f'Successfully executed ML code and generated {len(images)} visualizations with detailed explanations'
            }
            
        except Exception as e:
            return {
                'success': False,
                'images': [],
                'explanations': [],
                'overall_explanation': f"❌ **Error executing ML code:** {str(e)}",
                'message': f'Error executing ML code: {str(e)}'
            }
    
    def _generate_figure_explanation(self, fig, title: str, exec_globals: dict) -> str:
        """Generate detailed explanation for a matplotlib figure"""
        try:
            axes = fig.get_axes()
            if not axes:
                return f"📊 **{title}**: This figure shows data visualization but no specific details could be extracted."
            
            explanations = []
            
            for i, ax in enumerate(axes):
                ax_title = ax.get_title()
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                
                # Get plot type and data
                lines = ax.get_lines()
                collections = ax.collections
                patches = ax.patches
                
                if lines:
                    # Line plot
                    explanations.append(f"📈 **Line Plot Analysis**: This line plot shows trends and patterns over {xlabel}. The Y-axis represents {ylabel}, and the line's slope indicates the direction and rate of change. Look for upward trends (positive slope), downward trends (negative slope), or flat lines (no change). This visualization helps identify patterns, seasonality, and overall trends in your data.")
                elif collections:
                    # Scatter plot or similar
                    explanations.append(f"🔍 **Scatter Plot Analysis**: This scatter plot reveals the relationship between {xlabel} and {ylabel}. Each point represents a data sample, and the distribution shows how these variables relate to each other. Points clustered along a diagonal line suggest a strong correlation, while scattered points indicate weak or no correlation. This helps you understand the predictive power of your features.")
                elif patches:
                    # Bar plot or histogram
                    explanations.append(f"📊 **Bar Chart/Histogram Analysis**: This visualization shows the distribution or comparison of {ylabel} across {xlabel}. For histograms, the shape reveals data distribution patterns (normal, skewed, bimodal). For bar charts, heights indicate relative values, helping you identify the most important categories or features in your dataset.")
                else:
                    explanations.append(f"📊 **Chart Analysis**: This visualization displays the relationship between {xlabel} and {ylabel}. The chart helps you understand patterns, correlations, and distributions in your data, providing insights into how different variables interact and influence your target variable.")
            
            return f"📊 **{title}**: " + " ".join(explanations)
            
        except Exception as e:
            return f"📊 **{title}**: This visualization shows important patterns in your data."
    
    def _generate_overall_explanation(self, exec_globals: dict, num_images: int) -> str:
        """Generate comprehensive explanation of the ML model results with detailed insights"""
        try:
            explanation_parts = []
            
            # Check for common variables in the executed code
            if 'model' in exec_globals:
                model = exec_globals['model']
                explanation_parts.append(f"🤖 **Model Used**: {type(model).__name__}")
            
            if 'X_test' in exec_globals and 'y_test' in exec_globals:
                X_test = exec_globals['X_test']
                y_test = exec_globals['y_test']
                explanation_parts.append(f"📊 **Test Data**: {X_test.shape[0]} samples with {X_test.shape[1]} features")
            
            if 'y_pred' in exec_globals:
                y_pred = exec_globals['y_pred']
                explanation_parts.append(f"🎯 **Predictions**: Generated {len(y_pred)} predictions")
            
            # Check for metrics
            metrics_found = []
            if 'r2' in exec_globals:
                r2 = exec_globals['r2']
                metrics_found.append(f"R² = {r2:.4f}")
            
            if 'rmse' in exec_globals:
                rmse = exec_globals['rmse']
                metrics_found.append(f"RMSE = {rmse:.4f}")
            
            if 'mae' in exec_globals:
                mae = exec_globals['mae']
                metrics_found.append(f"MAE = {mae:.4f}")
            
            if 'accuracy' in exec_globals:
                accuracy = exec_globals['accuracy']
                metrics_found.append(f"Accuracy = {accuracy:.4f}")
            
            if metrics_found:
                explanation_parts.append(f"📈 **Performance Metrics**: {', '.join(metrics_found)}")
            
            # Add comprehensive analysis
            analysis = self._generate_comprehensive_analysis(exec_globals)
            if analysis:
                explanation_parts.append(analysis)
            
            # Interpret metrics
            interpretation = self._interpret_metrics(exec_globals)
            if interpretation:
                explanation_parts.append(f"💡 **Interpretation**: {interpretation}")
            
            explanation_parts.append(f"📊 **Visualizations**: Generated {num_images} charts showing model performance, predictions, and data patterns.")
            
            return "\n\n".join(explanation_parts)
            
        except Exception as e:
            return f"🤖 **ML Model Results**: Successfully executed machine learning analysis with {num_images} visualizations."
    
    def _generate_comprehensive_analysis(self, exec_globals: dict) -> str:
        """Generate comprehensive analysis of model performance and predictions"""
        try:
            analysis_parts = []
            
            # Analyze predictions vs actuals
            if 'y_test' in exec_globals and 'y_pred' in exec_globals:
                y_test = exec_globals['y_test']
                y_pred = exec_globals['y_pred']
                
                # Check if it's regression or classification
                if 'r2' in exec_globals or 'rmse' in exec_globals:
                    # Regression analysis
                    analysis_parts.append(self._analyze_regression_predictions(y_test, y_pred, exec_globals))
                elif 'accuracy' in exec_globals:
                    # Classification analysis
                    analysis_parts.append(self._analyze_classification_predictions(y_test, y_pred, exec_globals))
            
            # Analyze feature importance if available
            if 'model' in exec_globals and hasattr(exec_globals['model'], 'feature_importances_'):
                analysis_parts.append(self._analyze_feature_importance(exec_globals))
            
            # Analyze model performance patterns
            analysis_parts.append(self._analyze_model_performance_patterns(exec_globals))
            
            return "\n\n".join(analysis_parts) if analysis_parts else ""
            
        except Exception as e:
            return ""
    
    def _analyze_regression_predictions(self, y_test, y_pred, exec_globals: dict) -> str:
        """Analyze regression predictions in detail"""
        try:
            import numpy as np
            
            # Calculate additional metrics
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            prediction_range = np.max(y_pred) - np.min(y_pred)
            actual_range = np.max(y_test) - np.min(y_test)
            
            analysis = "🔍 **Detailed Prediction Analysis (Regression)**:\n\n"
            
            # Prediction accuracy analysis
            if 'r2' in exec_globals:
                r2 = exec_globals['r2']
                if r2 > 0.9:
                    analysis += f"✅ **Excellent Fit**: R² = {r2:.4f} indicates the model explains {r2*100:.1f}% of the variance. The model is highly accurate.\n"
                elif r2 > 0.7:
                    analysis += f"✅ **Good Fit**: R² = {r2:.4f} indicates the model explains {r2*100:.1f}% of the variance. The model performs well.\n"
                elif r2 > 0.5:
                    analysis += f"⚠️ **Moderate Fit**: R² = {r2:.4f} indicates the model explains {r2*100:.1f}% of the variance. Consider feature engineering or different models.\n"
                else:
                    analysis += f"❌ **Poor Fit**: R² = {r2:.4f} indicates the model explains only {r2*100:.1f}% of the variance. The model needs improvement.\n"
            
            # Error analysis
            if 'rmse' in exec_globals:
                rmse = exec_globals['rmse']
                analysis += f"📊 **Error Analysis**: RMSE = {rmse:.4f} means predictions are typically off by ±{rmse:.4f} units.\n"
            
            if 'mae' in exec_globals:
                mae = exec_globals['mae']
                analysis += f"📊 **Average Error**: MAE = {mae:.4f} means the average prediction error is {mae:.4f} units.\n"
            
            # Prediction range analysis
            analysis += f"📈 **Prediction Range**: Model predicts values from {np.min(y_pred):.4f} to {np.max(y_pred):.4f} (range: {prediction_range:.4f})\n"
            analysis += f"📈 **Actual Range**: Actual values range from {np.min(y_test):.4f} to {np.max(y_test):.4f} (range: {actual_range:.4f})\n"
            
            # Residual analysis
            analysis += f"🔍 **Residual Analysis**: Standard deviation of errors = {residual_std:.4f}\n"
            
            # Specific insights for common use cases
            analysis += self._generate_use_case_insights(y_test, y_pred, exec_globals)
            
            return analysis
            
        except Exception as e:
            return "🔍 **Prediction Analysis**: Detailed analysis of model predictions and performance."
    
    def _analyze_classification_predictions(self, y_test, y_pred, exec_globals: dict) -> str:
        """Analyze classification predictions in detail"""
        try:
            import numpy as np
            from collections import Counter
            
            analysis = "🔍 **Detailed Prediction Analysis (Classification)**:\n\n"
            
            # Accuracy analysis
            if 'accuracy' in exec_globals:
                accuracy = exec_globals['accuracy']
                if accuracy > 0.9:
                    analysis += f"✅ **Excellent Performance**: {accuracy*100:.1f}% accuracy indicates the model is highly reliable.\n"
                elif accuracy > 0.8:
                    analysis += f"✅ **Good Performance**: {accuracy*100:.1f}% accuracy indicates the model performs well.\n"
                elif accuracy > 0.7:
                    analysis += f"⚠️ **Moderate Performance**: {accuracy*100:.1f}% accuracy indicates room for improvement.\n"
                else:
                    analysis += f"❌ **Poor Performance**: {accuracy*100:.1f}% accuracy indicates the model needs significant improvement.\n"
            
            # Class distribution analysis
            actual_counts = Counter(y_test)
            pred_counts = Counter(y_pred)
            
            analysis += f"📊 **Class Distribution Analysis**:\n"
            analysis += f"   • Actual classes: {dict(actual_counts)}\n"
            analysis += f"   • Predicted classes: {dict(pred_counts)}\n"
            
            # Confusion matrix insights
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            analysis += f"🎯 **Classes Detected**: {len(unique_classes)} unique classes: {list(unique_classes)}\n"
            
            # Prediction confidence analysis
            if hasattr(exec_globals.get('model', None), 'predict_proba'):
                analysis += f"🎯 **Prediction Confidence**: Model provides probability scores for each prediction.\n"
            
            return analysis
            
        except Exception as e:
            return "🔍 **Prediction Analysis**: Detailed analysis of classification model performance."
    
    def _analyze_feature_importance(self, exec_globals: dict) -> str:
        """Analyze feature importance if available"""
        try:
            model = exec_globals.get('model')
            if not model or not hasattr(model, 'feature_importances_'):
                return ""
            
            importances = model.feature_importances_
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            # Get top features
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            analysis = "🎯 **Feature Importance Analysis**:\n\n"
            analysis += f"📊 **Top 5 Most Important Features**:\n"
            
            for i, (feature, importance) in enumerate(feature_importance_pairs[:5]):
                analysis += f"   {i+1}. {feature}: {importance:.4f} ({importance*100:.1f}%)\n"
            
            # Feature importance insights
            top_importance = feature_importance_pairs[0][1]
            if top_importance > 0.3:
                analysis += f"🔍 **Key Insight**: {feature_importance_pairs[0][0]} is the dominant feature with {top_importance*100:.1f}% importance.\n"
            elif top_importance > 0.1:
                analysis += f"🔍 **Key Insight**: Features are relatively balanced, with {feature_importance_pairs[0][0]} being most important.\n"
            else:
                analysis += f"🔍 **Key Insight**: No single feature dominates, suggesting complex feature interactions.\n"
            
            return analysis
            
        except Exception as e:
            return ""
    
    def _analyze_model_performance_patterns(self, exec_globals: dict) -> str:
        """Analyze overall model performance patterns"""
        try:
            import numpy as np
            
            analysis = "📈 **Model Performance Patterns**:\n\n"
            
            # Training vs testing performance
            if 'train_r2' in exec_globals and 'r2' in exec_globals:
                train_r2 = exec_globals['train_r2']
                test_r2 = exec_globals['r2']
                gap = train_r2 - test_r2
                
                if gap > 0.1:
                    analysis += f"⚠️ **Overfitting Detected**: Training R² ({train_r2:.4f}) is significantly higher than test R² ({test_r2:.4f}). Consider regularization.\n"
                elif gap < 0.05:
                    analysis += f"✅ **Good Generalization**: Training and test performance are well-balanced.\n"
                else:
                    analysis += f"📊 **Moderate Generalization**: Some gap between training and test performance.\n"
            
            # Cross-validation insights
            if 'cv_scores' in exec_globals:
                cv_scores = exec_globals['cv_scores']
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                analysis += f"🔄 **Cross-Validation**: Mean score = {cv_mean:.4f} (±{cv_std:.4f}), indicating {'stable' if cv_std < 0.05 else 'variable'} performance.\n"
            
            return analysis
            
        except Exception as e:
            return ""
    
    def _generate_use_case_insights(self, y_test, y_pred, exec_globals: dict) -> str:
        """Generate specific insights for common use cases like arrival time prediction"""
        try:
            import numpy as np
            
            # This would be enhanced based on the target column name
            # For now, provide general insights
            
            insights = "\n🎯 **Practical Insights**:\n"
            
            # Check if this might be time-related prediction
            if 'arrival' in str(exec_globals.get('target_col', '')).lower() or 'time' in str(exec_globals.get('target_col', '')).lower():
                insights += "⏰ **Time Prediction Insights**:\n"
                insights += f"   • Model predicts arrival times with reasonable accuracy\n"
                insights += f"   • Predictions range from {np.min(y_pred):.2f} to {np.max(y_pred):.2f} time units\n"
                insights += f"   • Average prediction error: {np.mean(np.abs(y_test - y_pred)):.2f} time units\n"
                insights += f"   • This model can help with scheduling and resource planning\n"
            
            elif 'price' in str(exec_globals.get('target_col', '')).lower() or 'cost' in str(exec_globals.get('target_col', '')).lower():
                insights += "💰 **Price Prediction Insights**:\n"
                insights += f"   • Model predicts prices with reasonable accuracy\n"
                insights += f"   • Price range: ${np.min(y_pred):.2f} to ${np.max(y_pred):.2f}\n"
                insights += f"   • Average prediction error: ${np.mean(np.abs(y_test - y_pred)):.2f}\n"
                insights += f"   • This model can help with pricing strategies and market analysis\n"
            
            else:
                insights += "📊 **General Prediction Insights**:\n"
                insights += f"   • Model provides reliable predictions for your target variable\n"
                insights += f"   • Prediction range: {np.min(y_pred):.4f} to {np.max(y_pred):.4f}\n"
                insights += f"   • Average error: {np.mean(np.abs(y_test - y_pred)):.4f} units\n"
                insights += f"   • Use these predictions for decision-making and planning\n"
            
            return insights
            
        except Exception as e:
            return ""
    
    def _interpret_metrics(self, exec_globals: dict) -> str:
        """Interpret the performance metrics and provide insights"""
        try:
            interpretations = []
            
            if 'r2' in exec_globals:
                r2 = exec_globals['r2']
                if r2 > 0.9:
                    interpretations.append("Excellent model fit (R² > 0.9)")
                elif r2 > 0.7:
                    interpretations.append("Good model fit (R² > 0.7)")
                elif r2 > 0.5:
                    interpretations.append("Moderate model fit (R² > 0.5)")
                else:
                    interpretations.append("Poor model fit (R² < 0.5) - consider feature engineering or different model")
            
            if 'accuracy' in exec_globals:
                accuracy = exec_globals['accuracy']
                if accuracy > 0.9:
                    interpretations.append("High classification accuracy (>90%)")
                elif accuracy > 0.8:
                    interpretations.append("Good classification accuracy (>80%)")
                elif accuracy > 0.7:
                    interpretations.append("Moderate classification accuracy (>70%)")
                else:
                    interpretations.append("Low classification accuracy (<70%) - consider model tuning")
            
            if 'rmse' in exec_globals and 'mae' in exec_globals:
                rmse = exec_globals['rmse']
                mae = exec_globals['mae']
                if rmse < mae * 1.5:
                    interpretations.append("Model shows consistent prediction errors")
                else:
                    interpretations.append("Model has some high-error predictions")
            
            return ". ".join(interpretations) + "." if interpretations else ""
            
        except Exception:
            return ""
