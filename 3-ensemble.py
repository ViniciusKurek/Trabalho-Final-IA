import joblib
import os
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

data = joblib.load('./features.joblib')
X_train = data['training_data']
y_train = data['training_labels']
X_test = data['validation_data']
y_test = data['validation_labels'] 

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

classificadores = []

nomes_arquivos = os.listdir('./resultados/ensemble')

for i, nome_arquivo in enumerate(nomes_arquivos):
    clf = joblib.load(os.path.join('./ensemble', nome_arquivo))
    classificadores.append((f'clf_{i+1}', clf))

votes = VotingClassifier(
    estimators=classificadores, 
    voting='soft',
)

votes.fit(X_train, y_train_encoded)

start_time = time.perf_counter()
y_pred_encoded = votes.predict(X_test)
end_time = time.perf_counter()
print(f"Tempo para classificação: {end_time - start_time}" + "\n")

print(classification_report(y_test_encoded, y_pred_encoded, digits=4))

cm = confusion_matrix(y_test_encoded, y_pred_encoded)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.colorbar()

plt.title("Matriz de confusão do ensemble")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")

classes = encoder.classes_
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

# Escreve valores dentro das células
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')

plt.tight_layout()
plt.show()