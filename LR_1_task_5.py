import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Завантаження даних
df = pd.read_csv('data_metrics.csv')

# Додавання стовпців для прогнозованих міток з порогом 0.5
thresh = 0.5
df['predicted_RF'] = (df['model_RF'] >= thresh).astype(int)
df['predicted_LR'] = (df['model_LR'] >= thresh).astype(int)

# Визначення функцій для обчислення метрик

# Знаходження TP, FN, FP, TN
def find_TP(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))

# Функція для створення матриці помилок
def custom_confusion_matrix(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Функції для обчислення метрик
def custom_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def custom_recall_score(y_true, y_pred):
    TP, FN = find_TP(y_true, y_pred), find_FN(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def custom_precision_score(y_true, y_pred):
    TP, FP = find_TP(y_true, y_pred), find_FP(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def custom_f1_score(y_true, y_pred):
    recall = custom_recall_score(y_true, y_pred)
    precision = custom_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Перевірка метрик на Random Forest (RF) та Logistic Regression (LR)
print("\nCustom Confusion Matrix RF:\n", custom_confusion_matrix(df['actual_label'].values, df['predicted_RF'].values))
print("\nCustom Confusion Matrix LR:\n", custom_confusion_matrix(df['actual_label'].values, df['predicted_LR'].values))

print("\nAccuracy RF:", custom_accuracy_score(df['actual_label'].values, df['predicted_RF'].values))
print("Accuracy LR:", custom_accuracy_score(df['actual_label'].values, df['predicted_LR'].values))

print("\nRecall RF:", custom_recall_score(df['actual_label'].values, df['predicted_RF'].values))
print("Recall LR:", custom_recall_score(df['actual_label'].values, df['predicted_LR'].values))

print("\nPrecision RF:", custom_precision_score(df['actual_label'].values, df['predicted_RF'].values))
print("Precision LR:", custom_precision_score(df['actual_label'].values, df['predicted_LR'].values))

print("\nF1 Score RF:", custom_f1_score(df['actual_label'].values, df['predicted_RF'].values))
print("F1 Score LR:", custom_f1_score(df['actual_label'].values, df['predicted_LR'].values))

# Побудова ROC-кривих
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df['actual_label'].values, df['model_RF'].values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df['actual_label'].values, df['model_LR'].values)

auc_RF = roc_auc_score(df['actual_label'].values, df['model_RF'].values)
auc_LR = roc_auc_score(df['actual_label'].values, df['model_LR'].values)

# Візуалізація ROC-кривих
plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Підрахунок і вивід метрик при зміні порога
thresholds = [0.5, 0.25]
for thresh in thresholds:
    df['predicted_RF'] = (df['model_RF'] >= thresh).astype(int)
    print(f"\nScores with threshold = {thresh}")
    print("Accuracy RF:", custom_accuracy_score(df['actual_label'].values, df['predicted_RF'].values))
    print("Recall RF:", custom_recall_score(df['actual_label'].values, df['predicted_RF'].values))
    print("Precision RF:", custom_precision_score(df['actual_label'].values, df['predicted_RF'].values))
    print("F1 Score RF:", custom_f1_score(df['actual_label'].values, df['predicted_RF'].values))
