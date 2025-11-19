from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
import time
import sys

def test_baselines():
    baseline_folder = "./baseline"
    rf_classifier = RandomForestClassifier()
    rf_classifier.set_params(
        n_estimators=200,
        max_depth=None,
        max_features=0.5,
        class_weight='balanced'
    )
    for name in os.listdir(baseline_folder):
        if name.endswith(".joblib"):
            file = os.path.join(baseline_folder, name)
            data = joblib.load(file)

            xTrain = data['training_data']
            yTrain = data['training_labels']
            xValidation = data['validation_data']
            yValidation = data['validation_labels']

            encoder = LabelEncoder()
            yTrainEncoded = encoder.fit_transform(yTrain)
            yValidationEncoded = encoder.transform(yValidation)


            rf_classifier.fit(xTrain, yTrainEncoded)
            pred = rf_classifier.predict(xValidation)
            # print(f"Relatório final do Random Forest com parâmetros escolhidos, baseline: {name}")
            report_str = classification_report(yValidationEncoded, pred, target_names=encoder.classes_, digits=4)
            # print(report_str)
            cm = confusion_matrix(yValidationEncoded, pred)
            # print("Matriz de Confusão:\n", cm)

            # monta caminho do arquivo de resultados (um .txt por baseline)
            results = os.path.join(baseline_folder, name.replace('.joblib', '') + '_results.txt')

            # formata matriz de confusão com rótulos legíveis
            labels = encoder.classes_
            header = " " * 16 + " ".join(f"{lab:>10}" for lab in labels) + "\n"
            rows = ""
            for i, row in enumerate(cm):
                rows += f"{labels[i]:>15} " + " ".join(f"{int(v):10d}" for v in row) + "\n"

            # escreve tudo no arquivo
            with open(results, "w", encoding="utf-8") as f:
                f.write(f"Relatório final do Random Forest com parâmetros escolhidos, baseline: {name}\n\n")
                f.write(report_str + "\n")
                f.write("Matriz de Confusão:\n")
                f.write(header)
                f.write(rows)

            print(f"Resultados salvos em: {results}")

ENSEMBLE_DIR = './ensemble'
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

def model_tuning(filepath, model, param_grid, scoring=make_scorer(f1_score, average='weighted'), cv=10, top_n=5):
    start_time = time.perf_counter()

    # carrega os dados extraídos
    data = joblib.load('features.joblib')
    xTrain = data['training_data']
    yTrain = data['training_labels']
    xValidation = data['validation_data']
    yValidation = data['validation_labels']

    encoder = LabelEncoder()
    yTrainEncoded = encoder.fit_transform(yTrain)
    yValidationEncoded = encoder.transform(yValidation)

    gridSearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        refit=True
    )
    gridSearch.fit(xTrain, yTrainEncoded)

    # Extrai os top N resultados do GridSearch
    results = gridSearch.cv_results_
    scores = results['mean_test_score']
    params = results['params']
    stds = results.get('std_test_score', None)
    order = np.argsort(scores)[::-1]
    top_n = min(top_n, len(order))

    model_type = model.__class__.__name__

    print(f"\nTop {top_n} configurações do GridSearch para {model} (ordenadas por mean_test_score):")
    top_models = []
    for rank, idx in enumerate(order[:top_n], start=1):
        p = params[idx]
        s = scores[idx]
        sd = stds[idx] if stds is not None else 0.0
        print(f"\n#{rank}: score={s:.4f} (+/-{sd:.4f}) params={p}")

        # re-treina o estimador com esses parâmetros para obter o modelo ajustado
        est = clone(model).set_params(**p)
        est.fit(xTrain, yTrainEncoded)
        filename = f"{model_type}_rank_{rank}.joblib"
        joblib.dump(est, os.path.join(ENSEMBLE_DIR, filename))
        pred_est = est.predict(xValidation)

        # imprime relatório e matriz de confusão para este modelo
        # grava as mesmas mensagens em arquivo de resultados
        try:
            with open(filepath, "a", encoding="utf-8") as out:
                out.write("Estimador ajustado:\n")
                out.write(str(est) + "\n")
                out.write("Relatório de classificação:\n")
                out.write(classification_report(yValidationEncoded, pred_est, target_names=encoder.classes_, digits=4))
            
                cm_est = confusion_matrix(yValidationEncoded, pred_est)
                labels = encoder.classes_
                # Imprime matriz com rótulos (colunas = previstos, linhas = reais)
                out.write("\nMatriz de Confusão:\n")
                # Cabeçalho
                out.write(" " * 16 + " ".join(f"{lab:>10}" for lab in labels) + "\n")
                # Linhas com rótulos
                for i, row in enumerate(cm_est):
                    out.write(f"{labels[i]:>15} " + " ".join(f"{int(v):10d}" for v in row) + "\n")
        except Exception as e:
            print(f"Erro ao escrever em {filepath}: {e}")

        time_elapsed = time.perf_counter() - start_time
        with open(filepath, "a", encoding="utf-8") as out:
            out.write(f"\nTempo decorrido até agora: {time_elapsed:.2f} segundos\n")


results_folder = "./resultados"
def knn():
    filepath = os.path.join(results_folder, "knn.txt")
    knn_param_grid = {
        'n_neighbors': [1, 3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    model_tuning(model=KNeighborsClassifier(), param_grid=knn_param_grid, filepath=filepath)

def rf():
    filepath = os.path.join(results_folder, "rf.txt")
    forest_grid = {
        'random_state': [42],
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20], 
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': ['balanced', None]
    }
    model_tuning(model=RandomForestClassifier(), param_grid=forest_grid, filepath=filepath)


def svm():
    filepath = os.path.join(results_folder, "svm.txt")
    svm_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None]
    }
    model_tuning(model=SVC(), param_grid=svm_grid, filepath=filepath)


def mlp():
    filepath = os.path.join(results_folder, "mlp.txt")
    mlp_grid = {
        'max_iter': [200, 400],
        'hidden_layer_sizes': [(100,), (150, 50), (200, 100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    model_tuning(model=MLPClassifier(), param_grid=mlp_grid, filepath=filepath)

# Busca melhores configurações para cada classificador
#knn()
rf()
svm()
mlp()