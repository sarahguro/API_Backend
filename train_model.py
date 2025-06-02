import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression

# Load the dataset
df = pd.read_csv('csv/student_habits_performance.csv')
df = df.drop(columns=['student_id'])

# Split features and targets
X = df.drop(columns=['exam_score'])
y_reg = df['exam_score']  # For regression
y_cls = (df['exam_score'] >= 50).astype(int)  # For classification (pass/fail)

# Identify column types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop='first'), categorical_cols)
])

# 1. Regression model pipeline
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])
reg_pipeline.fit(X, y_reg)
joblib.dump(reg_pipeline, "model_reg.pkl")

# 2. Classification model pipeline
cls_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
cls_pipeline.fit(X, y_cls)
joblib.dump(cls_pipeline, "model_cls.pkl")

# 3. Feature selection pipeline
feature_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(score_func=f_regression, k=10))
])
feature_pipeline.fit(X, y_reg)
joblib.dump(feature_pipeline, "model_feature.pkl")

print("Models saved: model_reg.pkl, model_cls.pkl, model_feature.pkl")
