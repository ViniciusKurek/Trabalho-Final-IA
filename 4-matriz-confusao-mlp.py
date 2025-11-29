import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


data = joblib.load('./features.joblib')
X_train = data['training_data']
y_train = data['training_labels']
X_test = data['validation_data']
y_test = data['validation_labels']

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

mlp_model = joblib.load('./ensemble/MLPClassifier_rank_1.joblib')
y_pred_encoded = mlp_model.predict(X_test)

print(classification_report(y_test_encoded, y_pred_encoded, target_names=encoder.classes_, digits=4))

cm = confusion_matrix(y_test_encoded, y_pred_encoded)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Matriz de Confusão MLP")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Verdadeira")

classes = encoder.classes_
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')

plt.colorbar()
plt.tight_layout()
plt.show()
