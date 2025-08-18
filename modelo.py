import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

#Modelo

file_path = 'Folds5x2_pp.xlsx'
data = pd.read_excel(file_path)


print("First few rows of the dataset:")
print(data.head())

X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(max_depth=15, random_state=42, n_estimators=60)
rf.fit(X_train_scaled, y_train)

train_score = rf.score(X_train_scaled, y_train)
test_score = rf.score(X_test_scaled, y_test)
print(f"Training R^2 score: {train_score:.2f}")
print(f"Testing R^2 score: {test_score:.2f}")


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)


plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist(), show=False)
plt.savefig("summary_plot.png", bbox_inches='tight')
plt.close()


plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist(), plot_type='bar', show=False)
plt.savefig("detailed_summary_plot.png", bbox_inches='tight')
plt.close()


for feature in X.columns:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_test_scaled, feature_names=X.columns.tolist(), show=False)
    plt.savefig(f"dependence_plot_{feature}.png", bbox_inches='tight')
    plt.close()
    

instance_index = 19
instance_scaled = scaler.transform(X_test.iloc[[instance_index]])
shap_value_instance = explainer.shap_values(instance_scaled)
force_plot = shap.force_plot(explainer.expected_value[0], shap_value_instance[0], instance_scaled[0], feature_names=X.columns.tolist())
shap.save_html("force_plot_instance.html", force_plot)

os.makedirs("assets", exist_ok=True)
for file in ["summary_plot.png", "detailed_summary_plot.png"] + [f"dependence_plot_{f}.png" for f in X.columns]:
    if os.path.exists(file):
        try:
            os.replace(file, os.path.join("assets", file))
        except Exception:
            pass