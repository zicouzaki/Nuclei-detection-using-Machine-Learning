import numpy as np
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
import cv2
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data.csv', sep=',')
X = df.drop(['Label'], axis=1)
y = df['Label']
X_test = df.iloc[1,:-1]
#print(df.iloc[1,:-1])
x = np.array(X_test).reshape(1,-1)
model = DecisionTreeClassifier()
model.fit(X, y)
feature = model.tree_.feature
print(feature)
print(len(feature))

n_nodes = model.tree_.node_count
print(n_nodes)

img = misc.imread("colon_normal_rose100.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
h,w = img.shape
print(h,w)
i = 0
tab = []
n_fois = 0
for i in range(0,w-20):
    j = 0
    for j in range(0,h-20):
        rect = []
        carreau = img[j:j+20,i:i+20]
        carre = (carreau.flatten().reshape(1,-1))
        y_predits = model.predict(carre)
        if(y_predits == 1):
            rect.append(j)
            rect.append(i)
            if(len(tab)==0):
                tab.append(rect)
                n_fois += 1
            # else:
            #     if(j >= tab[n_fois-1][1] and i >= tab[n_fois-1][0]):
            tab.append(rect)
            n_fois += 1
            #j += 20

plt.figure()
for i in range(0,len(tab)):
    #cv2.rectangle(img,(tab[i][1],tab[i][0]),(tab[i][1]+20,tab[i][0]+20),(0,255,0),1)
    cv2.circle(img,(tab[i][1]+10,tab[i][0]+10), 1, (0,255,0), 1)
plt.imshow(img,cmap='gray')
plt.show()
