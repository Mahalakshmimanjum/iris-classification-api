import joblib


# Load the saved pipeline
pipeline = joblib.load(
    "ml/saved_model/iris_pipeline.pkl"
)


# Brand-new sample
sample = [[
    5.1,  # sepal length
    3.5,  # sepal width
    1.4,  # petal length
    0.2   # petal width
]]


# Make prediction without retraining
prediction = pipeline.predict(sample)
probabilities = pipeline.predict_proba(sample)


print("Predicted class:", prediction[0])
print("Prediction probabilities:", probabilities[0])