import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class MLScientistAgent:
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.models = {
            'regression': ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Network', 'Transformer'],
            'classification': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network', 'Transformer'],
            'clustering': ['K-Means', 'DBSCAN', 'Hierarchical', 'Neural Network'],
            'time_series': ['LSTM', 'GRU', 'Transformer', 'ARIMA', 'Prophet']
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
        
        # Identify potential target variables
        numeric_cols = analysis['numeric_columns']
        categorical_cols = analysis['categorical_columns']
        
        # For regression tasks
        if len(numeric_cols) > 1:
            analysis['target_candidates'].extend(numeric_cols)
            analysis['recommended_models'].extend(self.models['regression'])
            analysis['task_type'] = 'regression'
        
        # For classification tasks
        if len(categorical_cols) > 0:
            analysis['target_candidates'].extend(categorical_cols)
            analysis['recommended_models'].extend(self.models['classification'])
            if analysis['task_type'] == 'unknown':
                analysis['task_type'] = 'classification'
        
        # For clustering tasks
        if len(numeric_cols) >= 2:
            analysis['recommended_models'].extend(self.models['clustering'])
        
        # Remove duplicates
        analysis['recommended_models'] = list(set(analysis['recommended_models']))
        
        return analysis
    
    def generate_ml_code(self, df: pd.DataFrame, target_column: str, model_type: str, task_type: str) -> str:
        """Generate comprehensive ML/DL code for the specified model and task"""
        
        if model_type.lower() in ['neural network', 'transformer']:
            return self._generate_deep_learning_code(df, target_column, model_type, task_type)
        else:
            return self._generate_traditional_ml_code(df, target_column, model_type, task_type)
    
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
df = pd.read_csv('your_data.csv')  # Replace with actual data loading
print(f"Dataset shape: {{df.shape}}")

# Separate features and target
target_col = '{target_column}'
"""
        
        if task_type == 'regression':
            code += f"""
# For regression task
y = df[target_col].values
X = df.drop(columns=[target_col])

# Handle categorical variables
categorical_cols = {categorical_cols}
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
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {{epoch_loss/len(train_loader):.4f}}, Val Loss: {{val_loss.item():.4f}}')

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

def model_predict(X):
    X_tensor = torch.FloatTensor(scaler.transform(X))
    model.eval()
    with torch.no_grad():
        return model(X_tensor).numpy()

# Calculate permutation importance
perm_importance = permutation_importance(
    model_predict, X_test, y_test, n_repeats=10, random_state=42
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

# Handle categorical variables
categorical_cols = {categorical_cols}
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
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {{epoch_loss/len(train_loader):.4f}}, Acc: {{100*correct/total:.2f}}%, Val Loss: {{val_loss.item():.4f}}, Val Acc: {{val_accuracy:.2f}}%')

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

def model_predict(X):
    X_tensor = torch.FloatTensor(scaler.transform(X))
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

# Calculate permutation importance
perm_importance = permutation_importance(
    model_predict, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
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
df = pd.read_csv('your_data.csv')  # Replace with actual data loading
print(f"Dataset shape: {{df.shape}}")

# Separate features and target
target_col = '{target_column}'
y = df[target_col].values
X = df.drop(columns=[target_col])

# Handle categorical variables
categorical_cols = {categorical_cols}
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
    
    def execute_ml_code(self, code: str, df: pd.DataFrame) -> dict:
        """Execute ML code and return results with visualizations"""
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
            
            # Capture any generated figures
            images = []
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    images.append(img_base64)
                plt.close('all')  # Close all figures to free memory
            
            return {
                'success': True,
                'images': images,
                'message': f'Successfully executed ML code and generated {len(images)} visualizations'
            }
            
        except Exception as e:
            return {
                'success': False,
                'images': [],
                'message': f'Error executing ML code: {str(e)}'
            }
