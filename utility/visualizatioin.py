__author__ = 'Haohan Wang'

import numpy as np

from matplotlib import pyplot as plt

data = np.load('../representation/pretrain/MOSI/audioRep.npy')
#
# data = np.load('../data/MOSI/audioFeatures.npy')[16:25,:40]

plt.imshow(data)
plt.show()
# for i in range(1500):
#     print videoTr[i,:10]