import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import pandas as pd

#For creating dataset from Images

m= 1
height= 800
width= 800
threshold_value = 128

o_csv_b = 'dataset/black/black.csv'
o_csv_w = 'dataset/white/white.csv'

list_b= {'p':'p.jpg','r':'r.jpg','q':'q.jpg','k':'k.jpg','n':'n.jpg','b':'b.jpg'}
list_w= {'P':'pw.jpg','R':'rw.jpg','Q':'qw.jpg','K':'kw.jpg','N':'nw.jpg','B':'bw.jpg'}

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

for im in range (1, 121):
    imgpath = os.path.join('boardimages', f"{im}.png")
    image = cv2.imread(imgpath)

    lower_green = np.array([60, 60, 60]) 
    upper_green = np.array([255, 255, 255]) 
    mask = cv2.inRange(image, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_board = image[y:y+h, x:x+w]
    cv2.imwrite('cropped.png', cropped_board)
    img1c = cv2.imread('cropped.png')  
    gray = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
    board = cv2.resize(gray, (800, 800))
    board = Image.fromarray(board)
    board.save('board_test.jpg')
    board= Image.open('board_test.jpg')

    for n in range(1, 9):
        height1 = n*100
        h1= (n-1)*100
        for s in range (1, 9):
            width1 = s*100
            w1= (s-1)*100
            pic= board.crop((h1,w1, height1, width1)) 
            pic.save('dummy.jpg')
            pic = cv2.imread("dummy.jpg") 
            lower = np.array([0,0,0])
            upper = np.array([90,90,90])
            lower1 = np.array([248,248,248])
            upper1 = np.array([256,256,256])
            mask1 = cv2.inRange(pic, lower, upper)
            mask2 = cv2.inRange(pic, lower1, upper1)
            cmask = cv2.bitwise_or(mask1, mask2)
            output = cv2.bitwise_and(pic,pic, mask= cmask)
            kernel = np.ones((5,5),np.float32)/25
            dst = cv2.filter2D(output,-1,kernel)
            resized_image = cv2.resize(dst, (20, 20), interpolation=cv2.INTER_AREA)
            # pics= f'{m}.jpg'
            # output = Image.fromarray(resized_image)
            # output.save(pics)
            imageA = resized_image
            imageBG= cv2.imread(f'bg.jpg')
            if np.any(imageA > threshold_value):
                for key,img in list_w.items():
                    imageB = cv2.imread(img)
                    err = mse(imageA, imageB)
                    if err <= 900:
                        flattened_image = imageA.flatten()
                        label=key
                        row = flattened_image.tolist()
                        row.append(label)
                        dfw = pd.DataFrame([row])
                        dfw.to_csv(o_csv_w, mode='a', header=False, index=False)
                        o_pic_w = os.path.join('dataset/pics/white', f'{m}.jpg')
                        output1 = Image.fromarray(imageA)
                        output1.save(o_pic_w)
            elif mse(imageA, imageBG) < 50:
                continue
            else:
                for key,img in list_b.items():
                    imageB = cv2.imread(img)
                    err = mse(imageA, imageB)
                    if err <= 200:
                        flattened_image = imageA.flatten()
                        label=key
                        row = flattened_image.tolist()
                        row.append(label)
                        dfb = pd.DataFrame([row])
                        dfb.to_csv(o_csv_b, mode='a', header=False, index=False)
                        o_pic_b = os.path.join('dataset/pics/black', f'{m}.jpg')
                        output1 = Image.fromarray(imageA)
                        output1.save(o_pic_b)
            m+=1