import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from  boruta import BorutaPy
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Loading and Cleaning
# ----------------------------

# Reading the data file
df = pd.read_csv("Cardiac-Arrhythmia/arrhythmia.data", header=None)
df.rename(columns={df.columns[-1]: 'Y'}, inplace=True)
df.to_csv("arrhythmia.csv", index=False)

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Data Cleaning:
# Remove rows where Y is in rare labels: 7,8,14,15,16
rare_lbl = [7, 8, 14, 15, 16]
df = df[~df['Y'].isin(rare_lbl)]

# Remove columns with only one unique value
df = df.loc[:, df.nunique() > 1]

# Count missing values per column and remove columns with more than 15 missing values
missing_counts = df.isna().sum()
df = df.loc[:, missing_counts <= 15]

# Convert non-numeric columns to numeric (if possible)
for col in df.columns:
    if not np.issubdtype(df[col].dtype, np.number):
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------
# 2. Missing Value Imputation and Class Merging
# ----------------------------

# Impute missing values for specific angle columns with the column mean
for col in ['QRST_angle', 'T_angle', 'P_angle']:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

# Merging classes:
# Merge class 4 into class 3
df.loc[df['Y'] == 4, 'Y'] = 3

# Merge classes 5 and 6 into 4
df.loc[df['Y'].isin([5, 6]), 'Y'] = 4

# Merge classes 9 and 10 into 5
df.loc[df['Y'].isin([9, 10]), 'Y'] = 5

# Convert Y to a categorical variable
df['Y'] = df['Y'].astype('category')

# ----------------------------
# 3. Scaling the Data
# ----------------------------

# Scale all features except Y using StandardScaler
features = df.drop('Y', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['Y'] = df['Y'].values  # append target variable

# ----------------------------
# 4. Splitting Data into Training and Test Sets
# ----------------------------

X = df_scaled.drop('Y', axis=1)
y = df_scaled['Y']

# Stratified split to keep class proportions; 75% training, 25% testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y
)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
# ----------------------------
# 5. Boruta Feature Selection
# ----------------------------

# BorutaPy requires the target as integers
y_train_codes = y_train.cat.codes.values

# Initialize a RandomForestClassifier (used by Boruta)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=123)

# Initialize Boruta feature selector
feat_selector = BorutaPy(rf, n_estimators='auto', random_state=123)

# Fit Boruta on training data
feat_selector.fit(X_train.values, y_train_codes)

# Get selected feature names
selected_features = X_train.columns[feat_selector.support_].tolist()
print("Selected features by Boruta:", selected_features)

# Create new train and test sets with only the selected features
X_train_boruta = X_train[selected_features]
X_test_boruta = X_test[selected_features]

# For convenience, attach the target variable back into new DataFrames
df_train_boruta = X_train_boruta.copy()
df_train_boruta['Y'] = y_train
df_test_boruta = X_test_boruta.copy()
df_test_boruta['Y'] = y_test

# ----------------------------
# 6. Linear Discriminant Analysis (LDA)
# ----------------------------

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_boruta, y_train)
pred_lda = lda.predict(X_test_boruta)
# For multiclass AUC, use the predicted probabilities and convert categories to codes.
pred_lda_proba = lda.predict_proba(X_test_boruta)
auc_lda = roc_auc_score(y_test.cat.codes, pred_lda_proba, multi_class='ovr')
print("LDA Multiclass AUC:", auc_lda)

# ----------------------------
# 7. Random Forest
# ----------------------------

# Define parameter grid: 'n_estimators' similar to ntree and 'max_features' (mtry)
param_grid_rf = {
    'n_estimators': [700, 1000, 2000],
    'max_features': [4, 8, 12, 16]
}

rf_model = RandomForestClassifier(random_state=123)
rf_search = RandomizedSearchCV(rf_model, param_distributions=param_grid_rf, 
                               cv=5, n_iter=10, random_state=123, n_jobs=-1)
rf_search.fit(X_train_boruta, y_train)
pred_rf = rf_search.predict(X_test_boruta)
accuracy_rf = accuracy_score(y_test, pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# ----------------------------
# 8. Support Vector Machines (RBF and Linear)
# ----------------------------

# SVM with RBF kernel
param_grid_svm_rbf = {
    'C': [2**i for i in range(-5, 6)],
    'gamma': [2**i for i in range(-7, 5)]
}
svm_rbf = SVC(kernel='rbf', probability=True, random_state=123)
svm_rbf_search = GridSearchCV(svm_rbf, param_grid=param_grid_svm_rbf, cv=5, n_jobs=-1)
svm_rbf_search.fit(X_train_boruta, y_train)
pred_svm_rbf = svm_rbf_search.predict(X_test_boruta)
pred_svm_rbf_proba = svm_rbf_search.predict_proba(X_test_boruta)
auc_svm_rbf = roc_auc_score(y_test.cat.codes, pred_svm_rbf_proba, multi_class='ovr')
print("SVM RBF Multiclass AUC:", auc_svm_rbf)

# SVM with Linear kernel
param_grid_svm_linear = {
    'C': [2**i for i in range(-5, 16)]
}
svm_linear = SVC(kernel='linear', probability=True, random_state=123)
svm_linear_search = GridSearchCV(svm_linear, param_grid=param_grid_svm_linear, cv=5, n_jobs=-1)
svm_linear_search.fit(X_train_boruta, y_train)
pred_svm_linear = svm_linear_search.predict(X_test_boruta)
# (Evaluation metrics can be computed similarly to above if needed)

# ----------------------------
# 9. Decision Trees
# ----------------------------

# Using 'entropy' to mimic information gain (split = "information" in R)
dtree = DecisionTreeClassifier(criterion='entropy', random_state=123)
# A simple grid search over max_depth (tuneLength=40 in R roughly corresponds to exploring a range)
dtree_search = GridSearchCV(dtree, param_grid={'max_depth': range(1, 21)}, cv=5, n_jobs=-1)
dtree_search.fit(X_train_boruta, y_train)
pred_dtree = dtree_search.predict(X_test_boruta)
accuracy_dtree = accuracy_score(y_test, pred_dtree)
print("Decision Tree Accuracy:", accuracy_dtree)

# Plot the best decision tree
plt.figure(figsize=(12, 8))
plot_tree(dtree_search.best_estimator_, feature_names=X_train_boruta.columns, 
          class_names=[str(cls) for cls in y.unique()], filled=True)
plt.title("Decision Tree")
plt.show()

# ----------------------------
# 10. XGBoost
# ----------------------------

# For XGBoost, convert Y to 0-indexed integers
y_train_xgb = y_train.cat.codes.copy()
y_test_xgb = y_test.cat.codes.copy()

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=len(y_train.unique()),
    eval_metric='merror',
    random_state=123, 
    n_jobs=-1
)
# Define a parameter grid similar to R's tuning parameters.
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],  # 'eta' in R
    'max_depth': list(range(3, 11)),
    'n_estimators': list(range(3, 21)),        # 'nrounds' in R
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'reg_lambda': [0, 1, 2],                   # 'lambda' in R
    'gamma': [0]
}

xgb_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid_xgb, 
                                cv=5, n_iter=5, random_state=123, n_jobs=-1)
xgb_search.fit(X_train_boruta, y_train_xgb)
xgb_pred = xgb_search.predict(X_test_boruta)
cm_xgb = confusion_matrix(y_test_xgb, xgb_pred)
print("XGBoost Confusion Matrix:\n", cm_xgb)

# ----------------------------
# 11. K-Nearest Neighbors (KNN)
# ----------------------------

knn = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': list(range(1, 21))
}
knn_search = GridSearchCV(knn, param_grid=param_grid_knn, cv=5, n_jobs=-1)
knn_search.fit(X_train_boruta, y_train)
pred_knn = knn_search.predict(X_test_boruta)
cm_knn = confusion_matrix(y_test, pred_knn)
print("KNN Confusion Matrix:\n", cm_knn)
accuracy_knn = accuracy_score(y_test, pred_knn)
print("KNN Accuracy:", accuracy_knn)

# For AUC calculation, get predicted probabilities.
pred_knn_proba = knn_search.predict_proba(X_test_boruta)
auc_knn = roc_auc_score(y_test.cat.codes, pred_knn_proba, multi_class='ovr')
print("KNN Multiclass AUC:", auc_knn)