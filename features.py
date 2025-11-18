import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from scipy.fft import fft2, ifft2

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

# Matiz será dividida em hBins, Saturação em sBins
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

def extract_cnn_features(img_path):
    img = image.load_img(img_path, target_size=(128, 256))
    imgArray = image.img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray = preprocess_input(imgArray)
    baseModel = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    featureExtractor = Model(inputs=baseModel.input, outputs=baseModel.output)
    features = featureExtractor.predict(imgArray)
    return features.flatten()

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
    #features.extend(extract_cnn_features(imgPath))

    for imgSection in split_img(img):
        features.extend(extract_hsv_histogram(imgSection))
        # features.extend(extract_hog(imgSection))
        # features.extend(extract_lbp(imgSection))
        pass

    return features