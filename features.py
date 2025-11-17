import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

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
    huMomentsLog = -np.sign(huMomentsRaw) * np.log10(np.abs(huMomentsRaw))
    return huMomentsLog.flatten()

def extract_hog(img, pixes_per_cell=16, cells_per_block=2):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizedImg = cv2.resize(grayImg, (64, 128))
    hogFeatures = hog(resizedImg,
                    pixels_per_cell=(pixes_per_cell, pixes_per_cell),
                    cells_per_block=(cells_per_block, cells_per_block), 
                    transform_sqrt=False, 
                    feature_vector=True)
    return hogFeatures


def extract_lbp(img, P=16, R=3):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(grayImg, P, R, method='uniform')
    bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), 
                           bins=bins, 
                           range=(0, bins), 
                           density=True)
    return hist

# Matiz será dividida em hBins, Saturação em sBins
def extract_hsv_histogram(img, hBins=10, sBins=4):
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

def split_img(img):
    height = img.shape[0]
    cutPoint = height // 2
    upper = img[0:cutPoint, :]
    lower = img[cutPoint:, :]
    return lower, upper

def extract_features(imgPath):
    img = cv2.imread(caminho)
    features = []
    features.extend(extract_hu_moments(img))

    for imgSection in split_img(img):
        features.extend(extract_hog(imgSection))
        features.extend(extract_lbp(imgSection))
        features.extend(extract_hsv_histogram(imgSection))
    
    return features
    

caminho = 'simpsons/Train/bart015.bmp'
imagem = cv2.imread(caminho)

features = extract_features(caminho)

print(features)