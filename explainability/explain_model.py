import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from lime.lime_tabular import LimeTabularExplainer


# ==========================================
# 1. Load the trained model
# ==========================================

model = joblib.load("ml/saved_model/model.joblib")

print("Model loaded successfully!")


# ==========================================
# 2. Load Iris dataset
# ==========================================

iris = load_iris()

feature_names = iris.feature_names
class_names = iris.target_names


# ==========================================
# 3. Global Feature Importance
# ==========================================

importances = model.feature_importances_

print("\n==============================")
print("Feature Importance")
print("==============================")

for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")


# ==========================================
# 4. Create Feature Importance Chart
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(feature_names, importances)

plt.title("Iris Model - Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("explainability/feature_importance.png")

plt.show()

print("\nFeature importance chart saved successfully!")


# ==========================================
# 5. Create LIME Explainer
# ==========================================

explainer = LimeTabularExplainer(
    training_data=iris.data,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    random_state=42
)

print("\nLIME explainer created successfully!")


# ==========================================
# 6. LIME Explanation - Sample 1
# ==========================================

sample1 = iris.data[0]

prediction1 = int(model.predict([sample1])[0])

print("\n==============================")
print("LIME Explanation - Sample 1")
print("==============================")

print("Input:", sample1)

print(
    "Predicted class:",
    class_names[prediction1]
)


explanation1 = explainer.explain_instance(
    sample1,
    model.predict_proba,
    num_features=4,
    labels=[prediction1]
)


print("\nFeature contributions:")

for feature, weight in explanation1.as_list(
    label=prediction1
):
    print(f"{feature}: {weight:.4f}")


explanation1.save_to_file(
    "explainability/lime_sample_1.html"
)

print(
    "\nLIME explanation saved successfully:"
    " lime_sample_1.html"
)


# ==========================================
# 7. LIME Explanation - Sample 2
# ==========================================

sample2 = iris.data[100]

prediction2 = int(model.predict([sample2])[0])

print("\n==============================")
print("LIME Explanation - Sample 2")
print("==============================")

print("Input:", sample2)

print(
    "Predicted class:",
    class_names[prediction2]
)


explanation2 = explainer.explain_instance(
    sample2,
    model.predict_proba,
    num_features=4,
    labels=[prediction2]
)


print("\nFeature contributions:")

for feature, weight in explanation2.as_list(
    label=prediction2
):
    print(f"{feature}: {weight:.4f}")


explanation2.save_to_file(
    "explainability/lime_sample_2.html"
)

print(
    "\nLIME explanation saved successfully:"
    " lime_sample_2.html"
)

