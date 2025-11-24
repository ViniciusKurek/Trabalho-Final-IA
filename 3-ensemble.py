import joblib
import os
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = joblib.load('./features.joblib')
X_train = data['training_data']
y_train = data['training_labels']
X_test = data['validation_data']
y_test = data['validation_labels'] 

classificadores = []

nomes_arquivos = os.listdir('./ensemble')

for i, nome_arquivo in enumerate(nomes_arquivos):
    clf = joblib.load(os.path.join('./ensemble', nome_arquivo))
    classificadores.append((f'clf_{i+1}', clf))

votes = VotingClassifier(
    estimators=classificadores, 
    voting='hard',
)

votes.fit(X_train, y_train)

y_pred = votes.predict(X_test)

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred))

print("\n--- Matriz de Confusão ---")
print(confusion_matrix(y_test, y_pred))