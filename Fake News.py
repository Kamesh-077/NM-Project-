import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# To upload a CSV file
from google.colab import files
uploaded = files.upload()

# Load the dataset
file_path = 'fake.csv'
df = pd.read_csv(file_path)

# Show basic info
print("First few rows:\n", df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# --- Visual 1: Missing Values Heatmap ---
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Drop missing values
df.dropna(inplace=True)

# --- Visual 2: Data Distribution ---
df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# Assume the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical features if needed
X = pd.get_dummies(X)
if y.dtype == 'object':
    y = pd.factorize(y)[0]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Visual 3: Class Balance ---
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Target Class Distribution")
plt.show()

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Visual 4: Confusion Matrix ---
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Visual 5: Feature Importances ---
importances = model.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df.head(15))
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
