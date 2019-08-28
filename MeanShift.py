import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_laplace, laplace, sobel, gaussian_filter
from scipy.interpolate import interp2d
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2hsv
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
image_file = 'tower.jpg'
img_rgb = np.array(Image.open(image_file))
height, width, channel = img_rgb.shape
hsv_img = rgb2hsv(img_rgb)
X = []
with tqdm(total=height*width) as pbar:
    for i in range(height):
        for j in range(width):
            X.append([0.05*i,0.05*j,img_rgb[i,j,0],img_rgb[i,j,1],img_rgb[i,j,2],hsv_img[i,j,0],hsv_img[i,j,1],hsv_img[i,j,2]])
            pbar.update(1)
# #############################################################################
# Compute clustering with MeanShift 0.1*i,0.05*j,
# The following bandwidth can be automatically detected using
X = np.array(X)
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
print(bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

segmented_image = np.array(labels).reshape(height,width)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
# #############################################################################
# # Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
