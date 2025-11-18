from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os
import numpy as np

dataPath = "simpsons"
trainPath = os.path.join(dataPath, "train")
validPath = os.path.join(dataPath, "valid")

data = {}
for j, dataType in enumerate([trainPath, validPath]):
    features = []
    labels = []
    
    for category in os.listdir(dataType):
        categoryPath = os.path.join(dataType, category)

        for imgName in os.listdir(categoryPath):
            imgPath = os.path.join(categoryPath, imgName)
            imgFeatures = extract_features(imgPath)
            features.append(imgFeatures)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

xTrain = np.array(data['training_data'])
yTrain = data['training_labels']
xValidation = np.array(data['validation_data'])
yValidation = data['validation_labels']

encoder = LabelEncoder()
yTrainEncoded = encoder.fit_transform(yTrain)
yValidationEncoded = encoder.transform(yValidation)

# RANDOM FOREST

model = RandomForestClassifier(random_state=42)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20], 
#     'max_features': ['sqrt', 'log2', 0.5],
#     'class_weight': ['balanced', None]
# }

# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [None], 
#     'max_features': [0.5],
#     'class_weight': ['balanced']
# }


# scorer = make_scorer(f1_score, average='weighted') 
# gridSearch = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring=scorer,
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# gridSearch.fit(xTrain, yTrainEncoded) 
# bestModel = gridSearch.best_estimator_
# pred = bestModel.predict(xValidation) 

# print("Melhor random forest encontrado:")
# print(bestModel)
# print("F1 score da validação cruzada: {:.4f}".format(gridSearch.best_score_))

# print(classification_report(yValidationEncoded, pred, target_names=encoder.classes_))

# cm = confusion_matrix(yValidationEncoded, pred) 
# print("Matriz de Confusão:\n", cm)

model = RandomForestClassifier(random_state=42, max_depth=None, n_estimators=200, max_features=0.5, class_weight='balanced')
# model = SVC(
#     random_state=109, 
#     C=1.0, 
#     kernel='rbf', 
#     class_weight='balanced',
#     gamma='auto' # Define como o kernel 'rbf' é calculado. 'scale' é o padrão recomendado.
# )
model.fit(xTrain, yTrainEncoded)
pred = model.predict(xValidation)
print(classification_report(yValidationEncoded, pred, target_names=encoder.classes_, digits=4))