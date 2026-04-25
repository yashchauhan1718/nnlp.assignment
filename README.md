# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ===================== LOAD DATA =====================
df = pd.read_csv("StudentsPerformance.csv")

# ===================== CLEAN COLUMNS =====================
df.columns = df.columns.str.strip().str.lower()

# ===================== REMOVE NULL =====================
df = df.dropna()

# ===================== ENCODE =====================
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# ===================== SELECT MATH COLUMN =====================
# CHECK printed columns if needed
print("Columns:", df.columns.tolist())

math_col = df.columns[5]   # 👉 change index if needed
print("Using:", math_col)

# ============================================================
# 🔴 PART 1: CLASSIFICATION (FIRST)
# ============================================================

# create classification target
df['pass'] = df[math_col].apply(lambda x: 1 if x >= 40 else 0)

X = df.drop([math_col, 'pass'], axis=1)
y_class = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

# ANN Classification
clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
clf.fit(X_train, y_train)

print("\n=== CLASSIFICATION RESULT ===")
print("Accuracy:", clf.score(X_test, y_test))

# ============================================================
# 🔵 PART 2: REGRESSION (SECOND)
# ============================================================

y_reg = df[math_col]

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2)

# ANN Regression
reg = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
reg.fit(X_train, y_train)

mae = np.mean(abs(reg.predict(X_test) - y_test))

print("\n=== REGRESSION RESULT ===")
print("MAE:", mae)

# ============================================================
# 🟢 HYPERPARAMETER TUNING
# ============================================================

reg2 = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=800)
reg2.fit(X_train, y_train)

mae2 = np.mean(abs(reg2.predict(X_test) - y_test))

print("\n=== AFTER TUNING ===")
print("New MAE:", mae2)

# ============================================================
# 📊 VISUALIZATION
# ============================================================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()# nnlp.assignment
