# ===================== IMPORTS =====================
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===================== LOAD DATA =====================
df = pd.read_csv("StudentsPerformance.csv")

# ===================== CLEAN =====================
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns.tolist())

# ===================== ENCODE =====================
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# ===================== SELECT TARGET COLUMN =====================
math_col = df.columns[5]   # 🔁 change if needed
print("Using:", math_col)

# ===================== CREATE CLASSIFICATION TARGET =====================
df['pass'] = df[math_col].apply(lambda x: 1 if x >= 40 else 0)

# ===================== FEATURES =====================
X = df.drop([math_col, 'pass'], axis=1).values
y_class = df['pass'].values
y_reg = df[math_col].values

# ===================== SPLIT =====================
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2
)

# ===================== TENSOR CONVERSION =====================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

y_class_train = torch.tensor(y_class_train, dtype=torch.float32).view(-1,1)
y_class_test  = torch.tensor(y_class_test,  dtype=torch.float32).view(-1,1)

y_reg_train = torch.tensor(y_reg_train, dtype=torch.float32).view(-1,1)
y_reg_test  = torch.tensor(y_reg_test,  dtype=torch.float32).view(-1,1)

# ===================== MODEL =====================
class MultiTaskANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared1 = nn.Linear(input_dim, 64)
        self.shared2 = nn.Linear(64, 32)

        self.class_out = nn.Linear(32, 1)  # classification
        self.reg_out   = nn.Linear(32, 1)  # regression

    def forward(self, x):
        x = torch.relu(self.shared1(x))
        x = torch.relu(self.shared2(x))

        class_output = torch.sigmoid(self.class_out(x))
        reg_output   = self.reg_out(x)

        return class_output, reg_output

model = MultiTaskANN(X_train.shape[1])

# ===================== LOSS + OPTIMIZER =====================
criterion_class = nn.BCELoss()
criterion_reg   = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===================== TRAIN =====================
for epoch in range(100):
    class_pred, reg_pred = model(X_train)

    loss_class = criterion_class(class_pred, y_class_train)
    loss_reg   = criterion_reg(reg_pred, y_reg_train)

    loss = loss_class + loss_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===================== TEST =====================
with torch.no_grad():
    class_pred, reg_pred = model(X_test)

    # classification accuracy
    class_pred = (class_pred > 0.5).float()
    acc = (class_pred == y_class_test).float().mean()

    # regression MAE
    mae = torch.mean(torch.abs(reg_pred - y_reg_test))

print("\nClassification Accuracy:", acc.item())
print("Regression MAE:", mae.item())
