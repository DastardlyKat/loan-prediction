from data_preprocessing import X_train, y_train, X_test, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

# ------------ Random Forest ------------ 
rf_clf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Classifier\n")
print(f'Accuracy Score: {rf_acc}')

joblib.dump(rf_clf, "loan_model.pkl")
print("Model saved as loan_model.pkl")