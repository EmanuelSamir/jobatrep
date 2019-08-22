import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import datetime
import logging
import pandas as pd
# from skimage.color import rgb2gray
import colorsys
from skimage.feature import greycomatrix, greycoprops

LEFTCLICK = 1
RIGHTCLICK = 3
NIMAGES = 15

class PixelPicking:
    def __init__(self, image,ax, pointsCounter, Count):
        self.image = image
        self.Count = Count
        self.ax = ax
        self.color_rightclick = 'red'
        self.color_leftclick = 'green'
        self.xs_Tower = []
        self.ys_Tower = []
        self.xs_NoTower = []
        self.ys_NoTower = []
        self.cidmouse = image.figure.canvas.mpl_connect('button_press_event', self.mouseclickCall)
        self.cidkey = image.figure.canvas.mpl_connect('key_press_event', self.keypressCall)
        self.pointsCounter = pointsCounter
        self.clicklst = []
        self.shape = {}
        self.sctlst = []

    def mouseclickCall(self, event):
        if event.button == LEFTCLICK:       # Tower
            # Append coordinates
            self.xs_Tower.append(event.xdata)
            self.ys_Tower.append(event.ydata)
            sct = self.ax.scatter(self.xs_Tower[-1],self.ys_Tower[-1],s=10,color=self.color_leftclick)
            self.sctlst.append(sct)
            self.pointsCounter += 1
            self.clicklst.append(LEFTCLICK)
            self.ax.set_title('Image #' +  str(self.Count) + "  Total points: "+ str(self.pointsCounter))
            self.image.figure.canvas.draw()

        if event.button == RIGHTCLICK:       # Tower
            # Append coordinates
            self.xs_NoTower.append(event.xdata)
            self.ys_NoTower.append(event.ydata)
            sct = self.ax.scatter(self.xs_NoTower[-1],self.ys_NoTower[-1],s=10,color=self.color_rightclick)
            self.sctlst.append(sct)
            self.pointsCounter += 1
            self.clicklst.append(RIGHTCLICK)
            self.ax.set_title('Image #' +  str(self.Count) + "  Total points: "+ str(self.pointsCounter))
            self.image.figure.canvas.draw()

    def keypressCall(self, event):
        if event.key == 'r' and self.pointsCounter >= 1:
            lastclick = self.clicklst[-1]
            if lastclick == LEFTCLICK:
                self.xs_Tower.pop(-1)
                self.ys_Tower.pop(-1)
            if lastclick == RIGHTCLICK:
                self.xs_NoTower.pop(-1)
                self.ys_NoTower.pop(-1)
            self.clicklst.pop(-1)
            self.pointsCounter -= 1
            self.sctlst[-1].remove()
            self.sctlst.pop(-1)
        self.ax.set_title('Image #' +  str(self.Count) + "  Total points: "+ str(self.pointsCounter))
        self.image.figure.canvas.draw()

    def recoverData(self):
        x_Tower, y_Tower, x_NoTower, y_NoTower = np.round(self.xs_Tower), np.round(self.ys_Tower), np.round(self.xs_NoTower), np.round(self.ys_NoTower)
        return x_Tower, y_Tower, x_NoTower, y_NoTower, self.pointsCounter


def pixels2HSV(image, pix_y, pix_x):
    h_params = []
    s_params = []
    v_params = []
    r_params = []
    g_params = []
    b_params = []
    for i, j in zip(pix_x, pix_y):
        x = colorsys.rgb_to_hsv(image[i,j,0]/255., image[i,j,1]/255., image[i,j,2]/255.)#rgb2hsv(np.expand_dims(np.expand_dims(image[i,j,:], axis=0), axis=0))
        #print(np.shape(x))
        h_params.append(x[0])
        s_params.append(x[1])
        v_params.append(x[2])
        r_params.append(image[i,j,0])
        g_params.append(image[i,j,1])
        b_params.append(image[i,j,2])
    return h_params, s_params, v_params, r_params, g_params, b_params

def pixels2GLCM(image, pix_y, pix_x, window = 15):
    distances = [2,3,4,5,6,7,8,9,10,11]
    angles = [0, np.pi/4, np.pi/2]
    contrast_params = []
    dissimilarity_params= []
    homogeneity_params= []
    ASM_params = []
    energy_params = []
    correlation_params = []

    gray = rgb2gray(image)
    for i, j in zip(pix_x, pix_y):
        # patch = rgb2gray(image[i-window:i+window, j-window:j+window,:].astype(float)).astype(int)
        patch = gray[i-window:i+window, j-window:j+window]
        glcm = greycomatrix(patch, distances, angles, 256)
        dissimilarity_params.append(greycoprops(glcm, 'dissimilarity'))#[0, 0])
        contrast_params.append(greycoprops(glcm, 'contrast'))#[0, 0])
        homogeneity_params.append(greycoprops(glcm, 'homogeneity'))#[0, 0])
        ASM_params.append(greycoprops(glcm, 'ASM'))#[0, 0])
        energy_params.append(greycoprops(glcm, 'energy'))#[0, 0])
        correlation_params.append(greycoprops(glcm, 'correlation'))#[0, 0])
    return dissimilarity_params, contrast_params, homogeneity_params, ASM_params, energy_params, correlation_params

def rgb2gray(rgb):
    x = np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.144]))
    gray = np.clip(x.astype('int'),0,255)
    return gray

def main():
    # Read all image filenames
    lst = os.listdir('.')
    imageslst = [k for k in lst if '.JPG' in k]
    random.shuffle(imageslst)
    # Variables to save data. Count = how many images. images = each image used. pointsCounter = pixels counted
    images = []
    Count = 0
    pointsCounter = 0
    # dfcombined = Complete dataframe to be used
    dfcombined = pd.DataFrame(columns = ['col-coord', 'row-coord', 'isTower', 'filename',
                                        'h_params', 's_params', 'v_params',
                                        'r_params', 'g_params', 'b_params',
                                        'dissimilarity_params','contrast_params', 'homogeneity_params','ASM_params', 'energy_params', 'correlation_params'])
    # For loop for first NIMAGES in shuffled image data
    for imageName in imageslst[0:NIMAGES]:
        Count += 1
        print('###########################################')
        print('######### Starting with #' + str(Count) + ' Image ########')
        print('########################################### \n')
        # Read image in list for each iter
        im = np.array(Image.open(imageName))
        # Open a figure for image plotting
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_title('Image #' +  str(Count) + "  Total point: " + str(pointsCounter) + '\n')
        currentImage = ax.imshow(im)
        # Function to save points through clicking manually
        #  Rigth click: isTower. Left Click: is not Tower. r: remove last point clicked
        pixelpicking = PixelPicking(currentImage, ax, pointsCounter, Count)
        ax.set_xlim(0,im[:,:,0].shape[1])
        ax.set_ylim(0,im[:,:,0].shape[0])
        plt.gca().invert_yaxis()
        # Show image
        plt.show()

        print('When enough points were marked, close the window to continue \n')
        images.append(im)
        # Recover data selected through clicking
        xs_Tower, ys_Tower, xs_NoTower, ys_NoTower, pointsCounter_ = pixelpicking.recoverData()
        print('Total of points recovered in this step are: ' + str(pointsCounter_ - pointsCounter) + '\n')
        pointsCounter = pointsCounter_
        # Close figure
        plt.close(f)
        print('Pixels selected are recovered and about to be processed \n')

        print('Saving data in a dataframe \n')
        # Creating dataframe and fill it with data alreadyknown
        df = pd.DataFrame(columns =['col-coord', 'row-coord', 'isTower'])
        df['col-coord'] = np.concatenate([xs_Tower, xs_NoTower]).astype(int)
        df['row-coord'] = np.concatenate([ys_Tower, ys_NoTower]).astype(int)
        df['isTower'] = (len(xs_Tower)*['Yes'] + len(xs_NoTower)*['No'])
        df['filename'] = (len(xs_Tower) + len(xs_NoTower)) * [imageName]
        # Add data from color maps: rgb and hsv
        print('HSV data collection \n')
        h_params, s_params, v_params, r_params, g_params, b_params = pixels2HSV(im, df['col-coord'].astype(int), df['row-coord'].astype(int))
        df['h_params'] = h_params
        df['s_params'] = s_params
        df['v_params'] = v_params
        df['r_params'] = r_params
        df['g_params'] = g_params
        df['b_params'] = b_params
        # Add data from GLCM arrays with patchs centered at pixels selected
        print('GLCM parameters collection \n')
        dissimilarity_params, contrast_params, homogeneity_params, ASM_params, energy_params, correlation_params = pixels2GLCM(im, df['col-coord'], df['row-coord'])
        df['dissimilarity_params'] = dissimilarity_params
        df['contrast_params'] = contrast_params
        df['homogeneity_params'] = homogeneity_params
        df['ASM_params'] = ASM_params
        df['energy_params'] = energy_params
        df['correlation_params'] = correlation_params

        print('Merging temporal dataframe from this step with global dataframe \n')
        dfcombined = pd.concat([dfcombined, df], ignore_index = True)

    print('All images were already processed\n')
    print(dfcombined[['col-coord','isTower','contrast_params','filename']])
    filename_csv = 'df' + str(datetime.date.today())+ '_n' +str(Count) +'_p' +str(pointsCounter) + '.pkl'
    dfcombined.to_pickle(r'./' + filename_csv)#, index = None, header=True)
    print('Dataframe saved in csv with name: ', filename_csv,'\n')


    print('Quitting')

if __name__ == "__main__":
    main()
