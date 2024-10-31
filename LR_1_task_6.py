import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utilities import visualize_classifier  # Переконайтеся, що utilities.py є в папці проекту

# Завантаження даних
input_file = 'data_multivar_nb.txt'  # Додайте цей файл у папку проекту
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення та тренування SVM-класифікатора
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X, y)

# Прогнозування за допомогою SVM-класифікатора
y_pred_svm = svm_classifier.predict(X)

# Оцінка якості класифікації для SVM
print("SVM Classifier:")
print("Accuracy:", accuracy_score(y, y_pred_svm))
print("Precision:", precision_score(y, y_pred_svm, average='weighted'))
print("Recall:", recall_score(y, y_pred_svm, average='weighted'))
print("F1 Score:", f1_score(y, y_pred_svm, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_svm))

# Візуалізація результатів SVM-класифікатора
visualize_classifier(svm_classifier, X, y)

# Створення та тренування наївного байєсовського класифікатора
nb_classifier = GaussianNB()
nb_classifier.fit(X, y)

# Прогнозування за допомогою наївного байєсовського класифікатора
y_pred_nb = nb_classifier.predict(X)

# Оцінка якості класифікації для наївного байєсовського класифікатора
print("\nNaive Bayes Classifier:")
print("Accuracy:", accuracy_score(y, y_pred_nb))
print("Precision:", precision_score(y, y_pred_nb, average='weighted'))
print("Recall:", recall_score(y, y_pred_nb, average='weighted'))
print("F1 Score:", f1_score(y, y_pred_nb, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_nb))

# Візуалізація результатів наївного байєсовського класифікатора
visualize_classifier(nb_classifier, X, y)

# Порівняння моделей
print("\nComparison of SVM and Naive Bayes Classifiers:")
print(f"SVM Accuracy: {accuracy_score(y, y_pred_svm)} vs Naive Bayes Accuracy: {accuracy_score(y, y_pred_nb)}")
print(f"SVM Precision: {precision_score(y, y_pred_svm, average='weighted')} vs Naive Bayes Precision: {precision_score(y, y_pred_nb, average='weighted')}")
print(f"SVM Recall: {recall_score(y, y_pred_svm, average='weighted')} vs Naive Bayes Recall: {recall_score(y, y_pred_nb, average='weighted')}")
print(f"SVM F1 Score: {f1_score(y, y_pred_svm, average='weighted')} vs Naive Bayes F1 Score: {f1_score(y, y_pred_nb, average='weighted')}")
