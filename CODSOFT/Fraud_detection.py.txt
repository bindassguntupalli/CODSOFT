import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

train_df = pd.read_csv('C:/Users/chara/Downloads/fraudDetection/fraudTrain.csv')
test_df = pd.read_csv('C:/Users/chara/Downloads/fraudDetection/fraudTest.csv')

numeric_columns = ['amt', 'city_pop', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long']

X_train = train_df[numeric_columns]
y_train = train_df['is_fraud']
X_test = test_df[numeric_columns]
y_test = test_df['is_fraud']

print("Missing values in training data:", X_train.isnull().sum())
print("Missing values in testing data:", X_test.isnull().sum())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train_res, y_train_res)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_res, y_train_res)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_res, y_train_res)

models = {'Logistic Regression': lr_model, 'Decision Tree': dt_model, 'Random Forest': rf_model}

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"--- {model_name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        print(f"ROC-AUC Score: {roc_auc}")
    print("\n")