import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\ACER\Dropbox\PC\Desktop\Jupyter-VS\Object Detection model\th.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stopdata = cv2.CascadeClassifier('stop_data.xml')
found = stopdata.detectMultiScale(img_gray, minSize=(20,20))

if found != 0:
    for (x, y, w, h) in found:
        cv2.rectangle(img_rgb, (x,y), x+h, y+w, (0,255,0), 5 )
        
plt.imshow(img_rgb)
plt.axis('off')
plt.show()