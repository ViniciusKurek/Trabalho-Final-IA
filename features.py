import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from img2vec_pytorch import Img2Vec
import joblib
from PIL import Image

from typing import List, Dict, Any


baseline_dir = os.path.join(".", "baseline2")
BASELINE = True

TRAIN = False

def extract_hu_moments(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaryImg = cv2.threshold(grayImg, 200, 255, cv2.THRESH_BINARY_INV)
    mask = np.zeros_like(binaryImg)
    contours, _ = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Preenche o personagem, considerando o maior contorno obtido
    # Interessante para preencher tudo que estiver dentro do contorno do personagem,
    # com o intuíto de melhorar o momento de Hu
    if contours:
        largerContour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largerContour], -1, 255, cv2.FILLED)
    
    moments = cv2.moments(mask)
    huMomentsRaw = cv2.HuMoments(moments)
    huMomentsLog = -np.sign(huMomentsRaw) * np.log10(np.abs(huMomentsRaw) + 1e-10)
    return huMomentsLog.flatten()

def extract_hog(img, pixelsPerCell=16, cellsPerBlock=2):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizedImg = cv2.resize(grayImg, (64, 64))
    hogFeatures = hog(resizedImg,
                    pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                    cells_per_block=(cellsPerBlock, cellsPerBlock), 
                    transform_sqrt=False, 
                    feature_vector=True)
    return hogFeatures

def extract_lbp(img, P=64, R=3):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(grayImg, P, R, method='uniform')
    bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), 
                           bins=bins, 
                           range=(0, bins), 
                           density=True)
    return hist

def extract_hsv_histogram(img, hBins=30, sBins=4):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsvImg],
        [0, 1], # Matiz, Saturação
        None, # Máscara, aplicar na imagem inteiro
        [hBins, sBins], 
        [0, 180, 0, 256] # Range dos canais
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def extract_glcm_features(img, distances =[1,3,5], angles = np.deg2rad([0,90,180,270])):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(grayImg, (128, 256))
    hists = graycomatrix(img, distances=distances, angles=angles, normed=True, symmetric=True)
    propNames = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    props = np.array([ graycoprops(hists, prop).flatten() for prop in propNames]).flatten()
    return props

def extract_img2vec_features(img, model):
    img2vec = Img2Vec(cuda=False, model=model)
    vec = img2vec.get_vec(img)
    return vec

def split_img(img):
    height = img.shape[0]
    cutPoint = height // 2
    upper = img[0:cutPoint, :]
    lower = img[cutPoint:, :]
    return lower, upper


def extract_features_baseline(imgPath):
    img = cv2.imread(imgPath)
    if img is None:
        return None
    feats = {}
    # global features
    feats['densenet161'] = np.asarray(extract_img2vec_features(img, 'densenet161')).flatten()
    feats['hu'] = np.asarray(extract_hu_moments(img)).flatten()
    feats['glcm'] = np.asarray(extract_glcm_features(img)).flatten()

    # section features (concatenate lower + upper for each feature type)
    hsv_parts = []
    hog_parts = []
    lbp_parts = []
    for section in split_img(img):
        hsv_parts.append(np.asarray(extract_hsv_histogram(section)).flatten())
        hog_parts.append(np.asarray(extract_hog(section)).flatten())
        lbp_parts.append(np.asarray(extract_lbp(section)).flatten())

    feats['hsv'] = np.concatenate(hsv_parts)
    feats['hog'] = np.concatenate(hog_parts)
    feats['lbp'] = np.concatenate(lbp_parts)

    # ensure all are plain numpy arrays
    for k in list(feats.keys()):
        feats[k] = np.asarray(feats[k]).flatten()

    return feats

def features_train(features):
    """
    Recebe uma lista de features (nomes ou caminhos para arquivos .joblib) gerados separadamente
    e concatena em um único arquivo joblib salvo em baseline_dir/combined_baseline.joblib.
    Retorna o caminho do arquivo gerado e o dicionário combinado.
    """
    # estrutura alvo
    combined = {
        'training_data': [],
        'validation_data': [],
        'training_labels': [],
        'validation_labels': []
    }

    if isinstance(features, str):
        features = [features]

    for feat in features:
        # determina caminho do arquivo .joblib
        if os.path.isfile(feat):
            path = feat
        else:
            fname = feat if feat.endswith('.joblib') else f"{feat}.joblib"
            path = os.path.join(baseline_dir, fname)

        if not os.path.exists(path):
            print(f"Aviso: arquivo de features não encontrado: {path}")
            continue

        try:
            loaded = joblib.load(path)
        except Exception:
            # pula arquivos que não puderam ser carregados
            continue

        # concatena safetly, esperando chaves padrão
        for key in combined.keys():
            val = loaded.get(key)
            if val is None:
                continue
            # aceita tanto listas quanto numpy arrays
            if isinstance(val, np.ndarray):
                combined[key].extend(val.tolist())
            else:
                combined[key].extend(list(val))

    # salva arquivo combinado
    out_name = "combined_baseline.joblib"
    out_path = os.path.join(baseline_dir, out_name)
    joblib.dump(combined, out_path)

    return out_path, combined

def main():
    dataPath = "simpsons"
    trainPath = os.path.join(dataPath, "train")
    validPath = os.path.join(dataPath, "valid")

    baseline_features_names = ['densenet161', 'hu', 'glcm', 'hsv', 'hog', 'lbp']

    # prepare storage structure for each feature (separate joblib files later)
    storage_baseline = {name: {'training_data': [], 'validation_data': [], 'training_labels': [], 'validation_labels': []}
            for name in baseline_features_names}
    
    # storage único (não separado por feature) — acumula vetores completos para treino/validação
    storage_train = {'training_data': [], 'validation_data': [], 'training_labels': [], 'validation_labels': []}

    # preenche storage por-feature (baseline) — comportamento original
    for j, dataType in enumerate([trainPath, validPath]):
        dataset_key = ['training_data', 'validation_data'][j]
        labels_key = ['training_labels', 'validation_labels'][j]

        for category in os.listdir(dataType):
            categoryPath = os.path.join(dataType, category)
            if not os.path.isdir(categoryPath):
                continue
        
            for imgName in os.listdir(categoryPath):
                imgPath = os.path.join(categoryPath, imgName)

                if BASELINE == True:
                    feats = extract_features_baseline(imgPath)
                    if feats is None:
                        continue

                    for name in baseline_features_names:
                        storage_baseline[name][dataset_key].append(feats[name].tolist())
                        storage_baseline[name][labels_key].append(category)

                    vec_baseline = np.concatenate([np.asarray(feats[name]).flatten() for name in baseline_features_names])

                    combined_storage_baseline[dataset_key].append(vec_baseline.tolist())
                    combined_storage_baseline[labels_key].append(category)


features_train(['./baseline/densenet161.joblib', './baseline/hsv.joblib'])