import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_laplace, laplace, sobel, gaussian_filter
from scipy.interpolate import interp2d
from PIL import Image
from tqdm import tqdm

N = 10
P = 16
SIGMA_START = 1
HBINS = 20
IMAGEFILE = 'Class_D.jpg'

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img(y,x) >= center:
            new_value = 1
    except:
        pass
    return new_value

def get_phase(L_x, L_y, x, y, radius):
    L_x_ = L_x[radius - SIGMA_START]
    L_y_ = L_y[radius - SIGMA_START]
    # Extract patch
    L_x_patch = L_x_[x-radius:x+radius+1,y-radius:y+radius+1]
    L_y_patch = L_y_[x-radius:x+radius+1,y-radius:y+radius+1]
    magnitude = (L_x_patch**2 + L_y_patch**2)**(1/2)
    yy, xx = np.meshgrid(np.arange(-radius,radius+1), np.arange(-radius,radius+1))
    distances = 1/2/np.pi/radius**2 * np.exp(-( ( (xx)**2 + (yy) **2)/2/radius**2))
    theta = np.arctan2(L_x_patch,L_y_patch)
    rows, cols = magnitude.shape
    hist, bin_edges = np.histogram(
                    theta.reshape(rows*cols,),
                    weights = (distances * magnitude).reshape(rows*cols,),
                    bins = np.linspace(-np.pi, np.pi, num = HBINS)
                    )
    return (np.diff(bin_edges)/2 + bin_edges[0:-1])[np.argmax(hist)]

def lbp_calculated_pixel(img, x, y, radius, L_x, L_y, th):
    x_pad = x + N
    y_pad = y + N

    theta = get_phase(L_x, L_y, x_pad, y_pad, radius)
    th.th.append(theta)
    xp_v = x + radius * np.cos(2*np.pi*np.arange(P)/P + theta)
    yp_v = y - radius * np.sin(2*np.pi*np.arange(P)/P + theta)
    center = img(y,x)
    val_ar = []
    for i in range(P):
        val_ar.append(get_pixel(img, center, xp_v[i], yp_v[i]))
    val = val_ar * 2**np.arange(P)
    return np.sum(val)

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "red":
            a = np.expand_dims(current_img,axis = 2)
            print(a)
            print(a.dtype)
            print(a.shape)
            b = (np.concatenate([a, np.zeros(a.shape, dtype = int),np.zeros(a.shape, dtype = int)], axis = 2))
            print(b)
            print(b.dtype)
            print(b.shape)
            current_plot.imshow(b)
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.hist(current_img, bins = np.linspace(0,2**P - 1, 2**(P-2)))
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)
    plt.show()

def get_scale(img):
    log_scales = []
    L_x = []
    L_y = []
    for sigma in np.arange(SIGMA_START,N):
        log_scales.append(gaussian_laplace(img, sigma = sigma, output = float))
        L_x.append( np.pad(
                sobel(gaussian_filter(img, sigma = sigma), axis = 1, output = float),
                ((N, N),(N, N)) ,  'reflect' , reflect_type = 'odd'
        ))
        L_y.append( np.pad(
                sobel(gaussian_filter(img, sigma = sigma), axis = 0, output = float),
                ((N, N),(N, N)) ,  'reflect' , reflect_type = 'odd'
        ))
    radius_array = np.argmax(np.array(log_scales), axis = 0) + SIGMA_START
    return radius_array, log_scales, L_x, L_y

class Angle:
    def __init__(self):
        self.th = []

def rgb2gray(rgb):
    x = np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.144]))
    gray = np.clip(x.astype('int'),0,255)
    return gray

def main():
    image_file = IMAGEFILE
    img_rgb = np.array(Image.open(image_file))
    height, width, channel = img_rgb.shape
    img_gray = img_rgb[:,:,0]#rgb2gray(img_rgb)
    img_gray_f = interp2d(np.arange(width), np.arange(height), img_gray)
    img_lbp = np.zeros((height, width), np.uint8)
    th = Angle()
    radius_array, log_scales, L_x, L_y = get_scale(img_gray)
    with tqdm(total=height*width) as pbar:
        for i in range(height):
            for j in range(width):
                 img_lbp[i, j] = lbp_calculated_pixel(img_gray_f, i, j, radius_array[i,j], L_x, L_y, th)
                 pbar.update(1)
    hist_lbp  = img_lbp.reshape(height*width,)
    h, b = np.histogram(radius_array.reshape(height*width,), bins = np.arange(N))
    print(h)
    print(b)
    print("---------")
    h, b = np.histogram(th.th, bins =  np.linspace(-np.pi, np.pi, num = HBINS))
    print(h)
    print(b)
    output_list = []
    output_list.append({
        "img": img_rgb,
        "xlabel": "xlabel",
        "ylabel": "ylabel",
        "xtick": [],
        "ytick": [],
        "title": "Coloured Image",
        "type": "gray"
    })
    output_list.append({
        "img": img_gray,
        "xlabel": "xlabel",
        "ylabel": "ylabel",
        "xtick": [],
        "ytick": [],
        "title": "Red Image",
        "type": "red"
    })
    output_list.append({
        "img": img_lbp,
        "xlabel": "xlabel",
        "ylabel": "ylabel",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })
    output_list.append({
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    })

    show_output(output_list)

    print("LBP Program is finished")

if __name__ == '__main__':
    main()
