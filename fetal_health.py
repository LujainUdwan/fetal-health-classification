import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/fetal_health.csv")

df.describe()
df.info()
df.isnull().sum()


# Correlation with target
nums = df.select_dtypes(include=[np.number])
corr = nums.corr()["fetal_health"]
print(corr)

corr = df.corr()

# Correlation heatmap
plt.figure(figsize=(24, 20))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.title("Correlation Matrix")
plt.show()

# Scatter plots: each feature vs target
target = "fetal_health"

for col in df.columns:
    if col == target:
        continue
    plt.figure(figsize=(5, 4))
    plt.scatter(df[col], df[target], s=10, alpha=0.6)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(col + " vs " + target)
    plt.show()
    plt.close()

# Feature distributions
plt.figure(figsize=(25, 15))
for i, column in enumerate(df.columns):
    plt.subplot(4, 6, i + 1)
    sns.histplot(data=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Class distribution pie chart
plt.figure(figsize=(10, 10))
plt.pie(
    df['fetal_health'].value_counts(),
    autopct='%.2f%%',
    labels=["NORMAL", "SUSPECT", "PATHOLOGICAL"],
    colors=sns.color_palette('Greys')
)
plt.title("Class Distribution")
plt.show()


# Prepare features and target
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define models
models_cls = [
    ("Logistic Regression", LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=0.01
    )),
    ("SVM (RBF)", SVC(
        kernel="rbf",
        C=10,
        gamma=0.01
    )),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )),
    ("Gradient Boosting", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.01,
        random_state=42
    )),
]

# Preprocessing pipeline
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


def run_all_classification_models(preprocess, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for name, model in models_cls:
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("="*50)
        print(name)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))


# Train on all features
run_all_classification_models(preprocess, X, y)


# Drop low-correlation histogram columns
cols_to_drop = [
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes"
]

X_dropped = df.drop(columns=cols_to_drop + ["fetal_health"])
y = df["fetal_health"]

print("New shape after dropping columns:", X_dropped.shape)

# Train on selected features
run_all_classification_models(preprocess, X_dropped, y)


# Voting classifier combining all models
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
                C=0.01
        )),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma=0.01,
            probability=True
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            random_state=42
        ))
    ],
    voting="soft"
)

voting_pipe_drop = Pipeline([
    ("preprocess", preprocess),
    ("model", voting_clf)
])

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dropped, y, test_size=0.2, random_state=42, stratify=y
)

voting_pipe_drop.fit(X_train_d, y_train_d)
preds_d = voting_pipe_drop.predict(X_test_d)

print("VOTING (After Feature Selection)")
print("Accuracy:", accuracy_score(y_test_d, preds_d))
print(classification_report(y_test_d, preds_d))


# Hyperparameter tuning — Logistic Regression
log_reg_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(
        solver="lbfgs",
        max_iter=2000
    ))
])
param_grid_lr = {"model__C": [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(log_reg_pipe, param_grid_lr, cv=5, scoring="accuracy", n_jobs=-1)
grid_lr.fit(X_dropped, y)
print("Best Logistic Regression Params:", grid_lr.best_params_)
print("Best CV Accuracy:", grid_lr.best_score_)


# Hyperparameter tuning — SVM
svm_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", SVC(kernel="rbf", probability=True))
])
param_grid_svm = {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]}
grid_svm = GridSearchCV(svm_pipe, param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1)
grid_svm.fit(X_dropped, y)
print("Best SVM Params:", grid_svm.best_params_)
print("Best CV Accuracy:", grid_svm.best_score_)


# Hyperparameter tuning — Random Forest
rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])
param_grid_rf = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}
grid_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1)
grid_rf.fit(X_dropped, y)
print("Best Random Forest Params:", grid_rf.best_params_)
print("Best CV Accuracy:", grid_rf.best_score_)


# Hyperparameter tuning — Gradient Boosting
gb_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])
param_grid_gb = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5]
}
grid_gb = GridSearchCV(gb_pipe, param_grid_gb, cv=5, scoring="accuracy", n_jobs=-1)
grid_gb.fit(X_dropped, y)
print("Best Gradient Boosting Params:", grid_gb.best_params_)
print("Best CV Accuracy:", grid_gb.best_score_)


# Cross-validation on final voting model
cv = KFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(
    voting_pipe_drop,
    X_dropped,
    y,
    cv=cv,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=-1
)
print("Train Accuracy:", res["train_score"].mean())
print("CV Accuracy:", res["test_score"].mean())
