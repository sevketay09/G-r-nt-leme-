# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:47:22 2021

@author: LENOVO
"""


### Resim okuma

import cv2
resim = cv2.imread('dag.jpg')
#print(resim)
cv2.imshow('dag',resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Gri resime çevirme

resim_gri = cv2.imread('dag.jpg',0)
cv2.imshow('gri_dag',resim_gri)
cv2.imwrite('gri_dag.jpg',resim_gri)

### Resmi yeniden boyutlandırma

# res = cv2.resize(resim,(480,480))

res = cv2.resize(resim,None,fx = 0.5,fy = 0.5)
cv2.imshow('resize_resim',res)
cv2.imwrite('resize_resim.jpg',res)

### Yakınlaştırma

res = cv2.resize(resim,None,fx = 2,fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('resize_resim',res)
cv2.waitKey(0)
cv2.destroyAllWindows()



### Uzaklaştırma

res = cv2.resize(resim,None,fx = 0.2,fy = 0.2, interpolation = cv2.INTER_AREA)
cv2.imshow('resize_resim',res)
cv2.waitKey(0)
cv2.destroyAllWindows()




### Resim özellikleri

print("Renkli resim : " + str(resim.shape))
print("Gri resim : " + str(resim_gri.shape))

print("Renkli resim size : " + str(resim.size))
print("Gri resim size : " + str(resim_gri.size))

print("Renkli resim type : " + str(resim.dtype))
print("Gri resim type : " + str(resim_gri.dtype))

### Çözünürlüğe göre pencere boyutu ayarlama

def main():
    img= cv2.imread('dag.jpg')
    ekran_cozunulurluk=1024,480

    skala_genislik=ekran_cozunulurluk[0]/img.shape[1]
    skala_yukseklik=ekran_cozunulurluk[1]/img.shape[0]
    skala=min(skala_genislik,skala_yukseklik)

    pencere_genislik=int(img.shape[1]*skala)
    pencere_yukseklik = int(img.shape[0] *skala)

    cv2.namedWindow('Boyutlanabilir Pencere',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Boyutlanabilir Pencere',pencere_genislik,pencere_yukseklik)

    cv2.imshow('Boyutlanabilir Pencere',img)
    cv2.imwrite('boyut_1.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
    



### Döndürme

# döndürme işlemi

sat,sut = resim_gri.shape

m = cv2.getRotationMatrix2D((sut/2,sat/2),180,1) # merkezden döndürmek için ikiye bölünür.
d = cv2.warpAffine(resim_gri,m,(sut,sat))

cv2.imshow("180 derece",d)
cv2.imwrite("180_derece_dondurulmus.jpg",d)
cv2.waitKey(0)
cv2.destroyAllWindows()

n = cv2.getRotationMatrix2D((sut/2,sat/2),90,1)
t = cv2.warpAffine(resim_gri,n,(sut,sat))

cv2.imshow("90 derece",t)
cv2.imwrite("90_derece_dondurulmus.jpg",t)
cv2.waitKey(0)
cv2.destroyAllWindows()


### Görüntünün transpozunu alma

rotated_image = cv2.transpose(resim)
cv2.imshow('transpose', rotated_image)
cv2.imwrite("transpose_resim.jpg",rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


### Fonksiyon ile görüntüyü sağa döndürme

import numpy as np

def sagaDondur(resim):
    
    resimT = resim.transpose() #transpoz alma
    katman,en,boy = np.shape(resimT) # transpoz aldığımızda dizilim de katman başa geliyor
    yeniResim = np.zeros((en,boy,katman), dtype=np.uint8) #tamamı 1 matris 
    for i in range(en-1):
        
        for j in range(boy-1):
            
            for k in range(katman):
                
                yeniResim[i,j,k] = resimT[k,i,boy-j-1]
    cv2.imshow("SagaDonmusResim",yeniResim)
    cv2.imwrite("saga_dondurulmus_resim.jpg",yeniResim)
    
sagaDondur(resim)


### Kırpma


height, width = resim.shape[:2]

start_row, start_col=int(height * .05), int(width * .30) # boyut isteğe bağlı değiştirilebilir.

end_row, end_col=int(height * .75), int(width * .75)

cropped=resim[start_row:end_row , start_col:end_col]

cv2.imshow('original', resim)

cv2.waitKey(0)

cv2.imshow('Kırpılmıs', cropped)
cv2.imwrite("kırpılmıs_resim.jpg",cropped)

cv2.waitKey(0)

cv2.destroyAllWindows()


### Öteleme

def translationalRight(image,count):
    
    resultImage=np.zeros((resim.shape[0]-count,resim.shape[1]-count,3),dtype=np.uint8)
    
    for i in range(resultImage.shape[0]):
        
        for j in range(resultImage.shape[1]):
            
            resultImage[i,j]=resim[i,j]
    return resultImage

def translationalLeft(image,count):
    resultImage=np.zeros((resim.shape[0]-count,resim.shape[1]-count,3),dtype=np.uint8)
    for i in range(resultImage.shape[0]):
        for j in range(resultImage.shape[1]):
            resultImage[i,j]=resim[count+i,count+j]
    return resultImage

def translationalUp(image,count):
    resultImage=np.zeros((resim.shape[0]-count,resim.shape[1]-count,3),dtype=np.uint8)
    for i in range(resultImage.shape[0]):
        for j in range(resultImage.shape[1]):
            resultImage[i,j]=resim[count+i,(int)(count/2)+j]
    return resultImage

def translationalDown(image,count):
    resultImage=np.zeros((resim.shape[0]-count,resim.shape[1]-count,3),dtype=np.uint8)
    for i in range(resultImage.shape[0]):
        for j in range(resultImage.shape[1]):
            resultImage[i,j]=resim[i,(int)(count/2)+j]
    return resultImage

resim = cv2.resize(resim,None,fx = 0.5,fy = 0.5)
cv2.imshow("Orginal Image",resim)
translationalImager=translationalDown(resim,50)
cv2.imshow("Translational Right Image",translationalImager)
translationalImagel=translationalDown(resim,50)
cv2.imshow("Translational Left Image",translationalImagel)
translationalImageu=translationalDown(resim,50)
cv2.imshow("Translational Up Image",translationalImageu)
translationalImaged=translationalDown(resim,50)
cv2.imshow("Translational Down Image",translationalImaged)
cv2.waitKey(0)


### Aynalama

aynalanmıs_resim = cv2.copyMakeBorder(res,75,75,125,125,cv2.BORDER_REFLECT)
cv2.imshow('Aynalanan Resim',aynalanmıs_resim)
cv2.imwrite('aynalanmis_resim.jpg',aynalanmıs_resim)
cv2.waitKey(0)

### Görüntü oluşturma

blank_image = np.zeros((height,width,3), np.uint8)
blank_image[:,0:width//2] = (255,0,0)      # (B, G, R)
blank_image[:,width//2:width] = (0,255,0)
cv2.imshow('Random Resim',blank_image)
cv2.imwrite('olusturulmus_resim.jpg',blank_image)

### Görüntü olusturma -2 

## Dikdörtgen ekleme

blank_image = cv2.rectangle(blank_image,(200,70),(320,180),(25,36,98),3)
cv2.imshow('dortgen',blank_image)
cv2.imwrite('olusturulmus_resim_2.jpg',blank_image)

### Görüntü oluşturma -3

## Çizgi ekleme

blank_image = cv2.line(blank_image,(200,70),(320,180),(25,36,98),3)
cv2.imshow('dortgen',blank_image)
cv2.imwrite('olusturulmus_resim_3.jpg',blank_image)

### Görüntü oluşturma -4

## Daire ekleme

blank_image = cv2.circle(blank_image,(360,360),50,(25,36,98),4)
cv2.imshow('dortgen',blank_image)
cv2.imwrite('olusturulmus_resim_4.jpg',blank_image)

# =============================================================================
#  NOKTA İŞLEMLERİ 
# =============================================================================

# Görüntü Negatifleme

import cv2
resim = cv2.imread('dag.jpg')
gray_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
imagem = (255-gray_resim)
cv2.imshow('Gri Resim',gray_resim)
cv2.imshow('terslenmiş resim',imagem)
cv2.imwrite('terslenmis_resim.jpg', imagem)
cv2.waitKey(0)

# Basit Eşikleme

import cv2
import numpy as np
from matplotlib import pyplot as plt
resim=cv2.imread('dag.jpg')
resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
ret, thresh1=cv2.threshold(resim,127,255,cv2.THRESH_BINARY)
ret, thresh2=cv2.threshold(resim,127,255,cv2.THRESH_BINARY_INV)
ret, thresh3=cv2.threshold(resim,127,255,cv2.THRESH_TRUNC)
ret, thresh4=cv2.threshold(resim,127,255,cv2.THRESH_TOZERO)
ret, thresh5=cv2.threshold(resim,127,255,cv2.THRESH_TOZERO_INV)

basliklar=['orjinal resim','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
resimler=[resim,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(resimler[i],'gray')
    plt.title(basliklar[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()

# Basit Eşikleme Fonksiyonu

import cv2
import numpy as np


def method1(img):
    """Numpy indexing"""
    img_thres = img
    img_thres[ img < 128 ] = 0
    return img_thres

def method2(img):
    """Double loop over pixels"""
    h = img.shape[0]
    w = img.shape[1]

    img_thres= np.zeros((h,w))
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            pixel = img[y, x]
            img_thres[y, x] = 0 if pixel < 128 else pixel
    return img_thres

resim = cv2.imread("dag.jpg")
resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orginal Image",resim)
thres_resim = method2(resim)
cv2.imwrite('esikleme_fonksiyon_method2.jpg', thres_resim)
cv2.imshow("Eşikleme",thres_resim)

cv2.waitKey(0)

# OTSU Segmentasyonu

img=cv2.imread("gurultuluresim.jpg",0)

_, th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('OTSU',th2)
cv2.imwrite('otsu_resim.jpg', th2)
cv2.waitKey(0)

# Logaritma Dönüşümü

import cv2
import numpy as np

img = cv2.imread('dag.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# log dönüşümü
img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255
img_log = np.array(img_log,dtype=np.uint8)

cv2.imshow('log_image',img_log)
cv2.imshow('original_img',img)
cv2.waitKey(0)


# Gamma Dönüşümü

import cv2
import numpy as np
  
# Open the image.
img = cv2.imread('dag.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.2, 2.2]:
      
    # Apply gamma correction.
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
  
    # Save edited images.
    cv2.imwrite('gamma_donusumu'+str(gamma)+'.jpg', gamma_corrected)


# Piecewise Linear Transformation

import cv2
import numpy as np
  
# Her yoğunluk seviyesini çıktı yoğunluğu seviyesine eşleme işlevi
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
  
# Open the image.
img = cv2.imread('dag.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Define parameters.
r1 = 70
s1 = 0
r2 = 140
s2 = 255
  
# Numpy dizisindeki her bir değere uygulamak için işlevi vektörleştirmek
pixelVal_vec = np.vectorize(pixelVal)
  
# Kontrast germe uygulayın
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
  
# 
cv2.imwrite('kontrast_germe.jpg', contrast_stretched)


# Histogram Eşitleme
# Hazır modül

import cv2
import numpy as np
  
img = cv2.imread('dag.jpg', 0)
  
equ = cv2.equalizeHist(img)

cv2.imshow('orjinal',img)
cv2.imshow('histogram', equ)
cv2.imwrite('histogram_esitleme.jpg', equ)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kümülatif Histogram

img = cv2.imread('dag.jpg', 0)

def make_histogram(image, bins=256):
    
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

def cumsum(values):
    result = [values[0]]
    for i in values[1:]:
        result.append(result[-1] + i)
    return result

def normalize(entries):
    
    numerator = (entries - np.min(entries)) * 255
    denorminator = np.max(entries) - np.min(entries)

    # re-normalize the cdf
    result = numerator / denorminator
    result.astype('uint8') # convert float into int

    return result

def equalizeHist(img):
    
    flatten_img = img.flatten() # 1D dönüşüm
    
    cumulativeSum = cumsum(make_histogram(flatten_img))
    
    cumulativeSum_norm = normalize(cumulativeSum)
    
    img_new_his = cumulativeSum_norm[flatten_img]
    
    # orijinal şekle dönüş
    img_new = np.reshape(img_new_his, img.shape)
    
    return img_new, cumulativeSum_norm


def drawImage(orignal, result, hist):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(orignal, cmap='gray')
    axes[0, 0].set_title('Kümülatif Histogram Eşitlemeden önce')
    
    axes[1, 0].hist(orignal.flatten(), 256, [0,256])
    
    
    axes[0, 1].imshow(result, cmap='gray')
    axes[0, 1].set_title('Kümalatif Histogram Eşitlemeden sonra')
    
    # Here you need to calculate hist of resulted image
    axes[1, 1].hist(result.ravel(),256,[0,256]);
    
    fig.savefig('sonuc.png')
    
result, normalized_cumsum = equalizeHist(img)
drawImage(img, result, normalized_cumsum)

# =============================================================================
# GÖRÜNTÜ ONARMA
# =============================================================================

# Gauss Gürültüsü ve Tuz Biber Gürültüsü

import numpy as np
import cv2

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      #row,col,ch= image.shape
      row = image.shape[0]
      col = image.shape[1]
      mean = 0
      var = 10
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
  
    
   elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.04
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out


  
img = cv2.imread('dag.jpg',0)
img = np.array(img/255, dtype=float)
noise_img = noisy("gauss",img)
noise_salt_pepper = noisy("s&p",img)
cv2.imshow('orjinal',img)
cv2.imshow('Gauss Gürültüsü', noise_img)
cv2.imwrite('gauss_noise_5.jpg', noise_img)
cv2.imshow('Tuz Biber Gürültüsü', noise_salt_pepper)
cv2.imwrite('s&p_noise_4.jpg', noise_salt_pepper)
  
cv2.waitKey(0)
cv2.destroyAllWindows()


### Filtreler

# Aritmetik Ortalama Filtresi

from matplotlib import pyplot as plt

img = cv2.imread('dag.jpg',0)
figure_size = 9
mean_filter = cv2.blur(img,(25, 25)) # 9*9'luk mask
plt.figure(figsize=(12,10))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(mean_filter, cmap='gray'),plt.title('25*25 Mean filter')
plt.xticks([]), plt.yticks([])
plt.show()


## Aritmetik Ortalama Filtresi Fonksiyon ile

import cv2
import numpy as np
  
     
# Read the image
img = cv2.imread('dag.jpg', 0)
 
# Obtain number of rows and columns
# of the image
m, n = img.shape
  
# Develop Averaging filter(3, 3) mask
mask = np.ones([3, 3], dtype = int)
mask = mask / 9
  
# Convolve the 3X3 mask over the image
img_new = np.zeros([m, n])
 
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
        
        img_new[i, j]= temp
         
img_new = img_new.astype(np.uint8)
cv2.imwrite('meanfilter_with_function.jpg', img_new)

# Geometric Mean Filter

img = cv2.imread('dag.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
rows, cols = img.shape[:2]
ksize = 5 # çekirdek boyutu

padsize = int((ksize-1)/2)
pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
geomean1 = np.zeros_like(img)
for r in range(rows):
    for c in range(cols):
        geomean1[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
geomean1 = np.uint8(geomean1)
cv2.imshow('geometric', geomean1)
cv2.imwrite('geofilter_with_function.jpg', geomean1)
cv2.waitKey()


# Gauss Filtresi

img = cv2.imread('dag.jpg',0)
gauss_img = cv2.GaussianBlur(img, (25,25),0) # 9*9'luk mask
plt.figure(figsize=(12,10))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gauss_img, cmap='gray'),plt.title('25*25 Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()

# Contraharmonic Mean Filtresi

img = cv2.imread('dag.jpg',0)

def contraharmonic_mean(img, size, Q):
    
    num = np.power(img, Q + 1)
    denom = np.power(img, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
    
    return result

resim = contraharmonic_mean(img, (3,3), 0.5)
cv2.imshow("contra",resim)
cv2.imwrite('contraharmonic_with_function.jpg', resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Median Filtresi

img = cv2.imread('dag.jpg',0)
median_img = cv2.medianBlur(img, 25)
plt.figure(figsize=(12,10))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(median_img, cmap='gray'),plt.title('25 - Median Filter')
plt.xticks([]), plt.yticks([])
plt.show()


## Median Filtresi Fonksiyon

img = cv2.imread('dag.jpg', 0)
 
m, n = img.shape
  

img_new1 = np.zeros([m, n])
 
for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [img[i-1, j-1],
               img[i-1, j],
               img[i-1, j + 1],
               img[i, j-1],
               img[i, j],
               img[i, j + 1],
               img[i + 1, j-1],
               img[i + 1, j],
               img[i + 1, j + 1]]
         
        temp = sorted(temp)
        img_new1[i, j]= temp[4]
 
img_new1 = img_new1.astype(np.uint8)
cv2.imwrite('medianfilter_with_function.jpg', img_new1)

# Midpoint Filter

from scipy.ndimage import maximum_filter, minimum_filter

img = cv2.imread('dag.jpg',0)

def midpoint(img):
    maxf = maximum_filter(img, (3, 3))
    minf = minimum_filter(img, (3, 3))
    midpo = (maxf + minf) / 2
    return midpo

mid = midpoint(img)
cv2.imwrite('midpointfilter.jpg', mid)


# =============================================================================
# Görüntü Onarma -2 
# =============================================================================

# Canny Kenar Bulma

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kedi.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100,200)
cv2.imshow("original",img)
cv2.imshow("canny",edges)
cv2.imwrite("canny.jpg",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Fonksiyon ile Canny

import numpy as np
import cv2
import matplotlib.pyplot as plt
 
  
# defining the canny detector function
  
# here weak_th and strong_th are thresholds for
# double thresholding step
def Canny_detector(img, weak_th = None, strong_th = None):
     
    # grayscale çevirme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    # Gürültü ekleme
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
      
    # Gradient hesabı
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
     
    # Kartezyen koordinatların kutuplara dönüştürülmesi
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
      
    # min - max threshold
    # iki kere threshold
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
     
    
    height, width = img.shape
      
    # her piksel için for döngüsü
    for i_x in range(width):
        for i_y in range(height):
              
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
              
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
             
            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
             
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
             
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
             
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
  
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
  
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)             
    ids = np.zeros_like(img)
      
    for i_x in range(width):
        for i_y in range(height):
             
            grad_mag = mag[i_y, i_x]
             
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
      
      
    return mag
  
frame = cv2.imread('kedi.png')
 
# calling the designed function for
# finding edges
canny_img = Canny_detector(frame)
  
# Displaying the input and output image 
cv2.imshow("canny",canny_img)
cv2.imwrite("canny_fonksiyon.jpg",canny_img)

# Sobel Filtresi

img = cv2.imread('kedi.png',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize = 3)

cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)
cv2.imwrite('sobelx.jpg',sobelx)
cv2.imwrite('sobely.jpg',sobely)


# =============================================================================
# MORFOLOJİK İŞLEMLER
# =============================================================================


# Erosion - Dilation

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('j.png')
img2=cv2.imread('j2.png')
img3=cv2.imread('j3.png')

kernel=np.ones((2,2),np.uint8)
erosion=cv2.erode(img2,kernel,iterations=1)
dilation=cv2.dilate(img3,kernel,iterations=1)

plt.subplot(321),plt.imshow(img),plt.title('orjinal')
plt.xticks([]),plt.yticks([])
plt.subplot(322),plt.imshow(img),plt.title('orjinal')
plt.xticks([]),plt.yticks([])
plt.subplot(323),plt.imshow(img2),plt.title('img2')
plt.xticks([]),plt.yticks([])
plt.subplot(324),plt.imshow(erosion),plt.title('erosion')
plt.xticks([]),plt.yticks([])
plt.subplot(325),plt.imshow(img3),plt.title('img3')
plt.xticks([]),plt.yticks([])
plt.subplot(326),plt.imshow(dilation),plt.title('dilation')
plt.xticks([]),plt.yticks([])
plt.show()

# Opening - Closing
# Dilation - Erosion

import cv2
import numpy as np

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imshow("erosion",erosion)
cv2.imwrite("erosion.jpg",erosion)


dilation = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow("dilation",dilation)
cv2.imwrite("dilation.jpg",dilation)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening",opening)
cv2.imwrite("opening.jpg",opening)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing",closing)
cv2.imwrite("closing.jpg",closing)

