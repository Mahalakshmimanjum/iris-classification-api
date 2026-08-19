from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Create the machine learning model
model = RandomForestClassifier(
    random_state=42
)


# 4. Train the model
model.fit(X_train, y_train)


# 5. Make predictions on test data
y_pred = model.predict(X_test)


# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")


# 7. Save the trained model
joblib.dump(model, "ml/saved_model/model.joblib")

print("Model saved successfully!")