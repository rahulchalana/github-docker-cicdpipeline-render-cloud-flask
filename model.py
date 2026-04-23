from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# Load data and train a tiny model
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# Save the model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as iris_model.pkl")