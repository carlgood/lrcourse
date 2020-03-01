import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_image_feature(p, s, q, k):
    #对于s保留前k个特征值
    k = int(k)
    s_temp = np.zeros(q.shape[0])
    s_temp[0:k] = s[0:k]
    E = np.eye(p.shape[0], q.shape[0])
    s = s_temp* E   #将k个特征值乘以奇异矩阵

    A_temp = np.dot(p, s)
    A_temp = np.dot(A_temp, q)
    plt.imshow(A_temp, cmap=plt.cm.gray, interpolation= 'nearest')
    plt.show()

if __name__ == "__main__":
    image = Image.open("./桌面.bmp").convert('L')
    A = np.array(image)
    print(A)

    plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

    p, s, q = np.linalg.svd(A, full_matrices=True, compute_uv= True)
    get_image_feature(p, s, q, k=0.01*min(s.shape))
    get_image_feature(p, s, q, k=0.05*min(s.shape))
    get_image_feature(p, s, q, k=0.5*min(s.shape))

