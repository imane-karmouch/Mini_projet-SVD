from PIL import Image 
import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
from skimage import exposure

#extraire l'image de lena grise
path="C:\\Users\HP\OneDrive\Documents\Lsd 2\Analyse de données\lena_gris.png"
im=Image.open(path)
im.show()
T=np.array(im)

#extraire l'image de lena en couleur
path2="C:\\Users\HP\OneDrive\Documents\Lsd 2\Analyse de données\lena_couleur.png"
im2=Image.open(path2)
im2.show()
C=np.array(im2)
C1=C[:,:,0]
C2=C[:,:,1]
C3=C[:,:,2]

#les differents composantes de la svd pour l'image grise
u, s, vt = alg.svd(T)

#la fonction compression qui prend en argument une matrice et un k et nous donne
#compressee et le taux de compreesion de k
def compression(M,k):
    u, s, vt = alg.svd(M)
    t=M.shape[0]*M.shape[1]
    a=[s[i] for i in range(k)]
    S=np.diag(a)
    U= u[: ,:k]
    Vt=vt[:k ,: ]
    Y=U@S@Vt
    T=(k*(M.shape[0]+1+M.shape[1]))/t
    return Y,T

#2eme methode pour la compression
"""def compression2(M,k):
    u, s, vt = alg.svd(M)
    for i in range(k):
        N= np.matrix(u[:, :i]) * np.diag(s[:i]) * np.matrix(vt[:i,:])
        
    print(N.shape)
    return N"""


#affichage des images compressees en gris avec les differents k demande
'''for j in range(10,160,10):
    Y=compression (T,j)[0]
    title = " Image after compression k =  %s" %j,'Taux de compression: %s' %compression (T,j)[1]
    plt.title(title)
    plt.imshow(Y, cmap='gray')
    plt.show()
    result = Image.fromarray((Y).astype(np.uint8))'''
    
#le graphe donnant le Taux de compression en fonction des valeurs singulière
'''x = [k for k in range(s.size)]  
y=[compression(T, k)[1] for k in range(s.size)]
plt.plot(x,y)
plt.show'''

#affichage des images compressees en couleur avec les differents k demande
for k in range(40,200,40):
    Y1=compression(C1,k)[0]
    Y2=compression(C2,k)[0]
    Y3=compression(C3,k)[0]
    
    D= np . zeros ( (435 ,395, 3 ) , dtype = np.uint8 )
    D[:,:,0]=Y1
    D[:,:,1]=Y2
    D[:,:,2]=Y3
    title = " Image after compression k =  %s" %k
    plt.title(title)
    plt.imshow(D)
    plt.show()
    
    
#Fonction qui calcule la valeur de k permettant de capter 95% de la variance     
'''def variance(s):
    v=0
    k=0
    d=sum(s)
    while (v<0.95):
        v=(sum(s[:k+1])/d)**2
        k+=1
    return k'''

#graphe des des valeurs singulière en fonction de k
'''
x = [k for k in range(s.size)]
y=[s[k] for k in range(s.size)]
plt.plot(x,y)
plt.show()'''


#graphe de la variance en fonction des k
'''
x1=[k for k in range(s.size)]
y1=[(sum(s[:k+1])/np.sum(s))**2 for k in range(s.size)]
plt.plot(x1,y1)
plt.show()'''

#graphe indiquant le niveau de gris cad  le degré de sombritude de l 'image grise
'''def imageHist(image):
    _, axis = plt.subplots(ncols=2, figsize=(12, 3))
    axis[0].imshow(image, cmap=plt.get_cmap('gray'))
    axis[1].set_title('Histogram')
    axis[0].set_title('Grayscale Image')
    hist = exposure.histogram(image)
    axis[1].plot(hist[0])
imageHist(T)'''
