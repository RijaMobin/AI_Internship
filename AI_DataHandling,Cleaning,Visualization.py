# -----------------------------------
# 📦 Step 1: Import Libraries
# -----------------------------------
import pandas as pd        # For data handling
import numpy as np         # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns      # For beautiful plots
import plotly.express as px   # For interactive plots
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression   # ML model
from sklearn.metrics import accuracy_score, confusion_matrix  # For evaluation

# -----------------------------------
# 📄 Step 2: Create Dummy Student Dataset
# -----------------------------------
data = {
    'Name': ['Ali', 'Sara', 'John', 'Ahmed', 'Mira', 'Asad', 'Zara', 'Omar'],
    'Math': [45, 78, 90, 30, 66, 38, np.nan, 80],      # Includes missing value
    'Science': [50, 82, 88, 35, 60, 42, 70, 90],
    'English': [48, 80, 85, 33, 58, 40, 65, 88]
}
df = pd.DataFrame(data)

# -----------------------------------
# 🧹 Step 3: Data Cleaning
# -----------------------------------
# Fill missing value with average of Math column
df['Math'].fillna(df['Math'].mean(), inplace=True)

# Create 'Average' column for all subjects
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

# Create 'Passed' column (label): 1 if average >= 50, else 0
df['Passed'] = (df['Average'] >= 50).astype(int)

# -----------------------------------
# 👁️ Step 4: Data Exploration / Visualization
# -----------------------------------
# Histogram of marks
sns.histplot(df['Average'], kde=True)
plt.title("Distribution of Average Marks")
plt.show()

# Boxplot to detect outliers
sns.boxplot(data=df[['Math', 'Science', 'English']])
plt.title("Boxplot of Subject Marks")
plt.show()

# Correlation Heatmap
sns.heatmap(df[['Math', 'Science', 'English', 'Average', 'Passed']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Interactive Scatter Plot (Plotly)
fig = px.scatter(df, x='Math', y='Science', color=df['Passed'].map({0: 'Fail', 1: 'Pass'}))
fig.update_layout(title="Math vs Science with Pass/Fail")
fig.show()

# -----------------------------------
# 🤖 Step 5: Train a Machine Learning Model
# -----------------------------------
# Features (X) = Subject Marks | Label (y) = Passed
X = df[['Math', 'Science', 'English']]
y = df['Passed']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# -----------------------------------
# 📈 Step 6: Evaluate the Model
# -----------------------------------
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------------
# 📊 Step 7: Show Results
# -----------------------------------
# Show final table with prediction
df['Prediction'] = model.predict(X)
print(df[['Name', 'Math', 'Science', 'English', 'Average', 'Passed', 'Prediction']])
