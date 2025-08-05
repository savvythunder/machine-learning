from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

# Load dataset
iris = load_iris() # <- Our dataset
X = iris.data #     Theta X [data]
y = iris.target #   Theta Y [target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 

# Train Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Make prediction
y_pred = log_reg.predict(X_test)
print("Logistic Regression model accuracy:", metrics.accuracy_score(y_test, y_pred))


def _():
    """A model accuracy of 0.96 indicates that your model correctly predicted the target variable 96% of the time on the test dataset. This is generally considered a very good accuracy, especially in many practical applications. However, it's important to consider a few additional points:

1. **Class Imbalance**: If your dataset has imbalanced classes, high accuracy might be misleading. For example, if 95% of your data belongs to one class, a model that always predicts that class would still achieve 95% accuracy.

2. **Other Metrics**: It's often useful to look at other evaluation metrics such as:
   - Precision: The ratio of true positive predictions to the total predicted positives.
   - Recall (Sensitivity): The ratio of true positive predictions to the total actual positives.
   - F1 Score: The harmonic mean of precision and recall, useful for imbalanced datasets.
   - Confusion Matrix: Provides a detailed breakdown of correct and incorrect predictions.

3. **Cross-Validation**: To ensure that your model's performance is consistent, consider using cross-validation. This technique helps to assess how the results of your model will generalize to an independent dataset.

4. **Overfitting**: A very high accuracy on the training set compared to the test set may indicate overfitting. Ensure that your model is not just memorizing the training data.

5. **Domain Knowledge**: Depending on the application, a 96% accuracy might be excellent or insufficient. Always consider the context of your problem.

If you want to further evaluate your model, you can calculate and print additional metrics as follows:


from sklearn.metrics import classification_report

# Print classification report
print(classification_report(y_test, y_pred))


This will give you a comprehensive overview of your model's performance across different metrics."""

# A activity example in which our model differenciate between label, e.g. - [CAT, DOG, BAT, DOG, CAT, BAT]
def label_differenciate():
    categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird'] # LIst a labels
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(categorical_feature)
    print("Encoded feature:", encoded_feature)