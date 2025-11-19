import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from img2vec_pytorch import Img2Vec
import joblib

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

def extract_features(imgPath):
    img = cv2.imread(imgPath)
    features = []
    #features.extend(extract_hu_moments(img))
    features.extend(extract_img2vec_features(img, model='densenet161'))

    for imgSection in split_img(img):
        features.extend(extract_hsv_histogram(imgSection))
        # features.extend(extract_hog(imgSection))
        # features.extend(extract_lbp(imgSection))
        pass
    
    return features


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

joblib.dump(data, "features.joblib");