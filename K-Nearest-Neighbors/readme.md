# K-Nearest Neighbors (KNN) Classifier on Iris Dataset

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier to classify iris plants into three species based on their features. The Iris dataset is a well-known dataset in machine learning, consisting of 150 samples with four features for each instance.

## Dataset
The Iris dataset contains the following features:

- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

### Sample Data
| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
|--------------------|-------------------|--------------------|-------------------|
| 5.1                | 3.5               | 1.4                | 0.2               |
| 4.9                | 3.0               | 1.4                | 0.2               |
| 4.7                | 3.2               | 1.3                | 0.2               |
| 4.6                | 3.1               | 1.5                | 0.2               |
| 5.0                | 3.6               | 1.4                | 0.2               |

## Model Training
The KNN classifier was trained on the dataset, achieving an accuracy of **100%**.

### Model Performance
The evaluation metrics for the model are as follows:

```
KNN Classifier Accuracy: 1.00

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

- **Precision**: The proportion of true positive results in the predicted positive results for each class.
- **Recall**: The proportion of true positive results in the actual positive results for each class.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

### Confusion Matrix
The confusion matrix indicates perfect classification across all classes.

## Conclusion
The KNN classifier provides a robust model for predicting iris plant species, achieving perfect accuracy on the test set. This indicates that the model was able to classify all instances correctly.

## Next Steps
To further enhance the model, consider the following:
- Experimenting with different values of `k` (number of neighbors).
- Implementing feature scaling (e.g., StandardScaler) to improve model performance.
- Using cross-validation to ensure the model's robustness.
- Exploring distance metrics (e.g., Euclidean, Manhattan) to see their impact on performance.

## License
This project is licensed under the MIT License.