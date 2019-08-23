from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
import random


img_width = 640
img_height = 480

min_n_circles = 30
max_n_circles = 66

n_images = 9300

min_width = 50
max_width = 101
min_height = 50
max_height = 101
min_depth = 50
max_depth = 101

min_x_offset = -500
max_x_offset = 501
min_y_offset = -500
max_y_offset = 501
min_z_offset = -500
max_z_offset = 501


max_prec_per_u = 101
max_prec_per_v = 101

perc_anomalies = 0.08

extra_colors = ["#D10D0D", "#FDFD0A", "#0A1AFD", "#88299F", "#88299F", "#F7540E", "#CCCCFF", "#36B346", "#11761F", "#14581D", 
                "#08F227", "#07A837", "#2D7342", "#7ED12B", "#4B6B2B", "#7AA252", "#167941", "#579271", "#A7E561", "#33FF33",
                "#009900", "#006633", "#66CC00", "#33FF99", "#006666", "#B2FF66", "#105826"]

green_colors = ["#36B346", "#11761F", "#14581D", "#08F227", "#07A837", "#2D7342", "#7ED12B", "#4B6B2B", "#7AA252", "#167941", 
                "#579271", "#A7E561", "#33FF33", "#009900", "#006633", "#66CC00", "#33FF99", "#006666", "#B2FF66", "#105826"]

def createEllipse(width, height, depth, x_offset, y_offset, z_offset, prec_per_u, prec_per_v):
    u = np.linspace(0, 2 * np.pi, prec_per_u)
    v = np.linspace(0, np.pi, prec_per_v)
    x = width * np.outer(np.cos(u), np.sin(v)) + x_offset
    y = height * np.outer(np.sin(u), np.sin(v)) + y_offset
    z = depth * np.outer(np.ones(np.size(u)), np.cos(v)) + z_offset
    return (x, y, z)

def generateImages(with_anomalies):
    it_start = 8772
    it_stop = n_images
    color_array = green_colors
    save_path = './dataset/no_anomalies/'
    min_prec_per_u = 50
    min_prec_per_v = 50
    if with_anomalies == True:
        it_start = n_images
        it_stop = n_images + int(n_images * perc_anomalies)
        min_prec_per_u = 1
        min_prec_per_v = 1
        color_array = extra_colors
        save_path = './dataset/anomalies/'
    for img_n in range(it_start, it_stop):
        rand_n_circles = random.randint(min_n_circles, max_n_circles)
        rand_n_circles = random.randint(min_n_circles, max_n_circles)
        fig = plt.figure(figsize=(img_width/100, img_height/100))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        ax.axis('off')
        for circle in range(rand_n_circles):
            width = random.randint(min_width, max_width)
            height = random.randint(min_height, max_height)
            depth = random.randint(min_depth, max_depth)
            x_offset = random.randint(min_x_offset, max_x_offset)
            y_offset = random.randint(min_y_offset, max_y_offset)
            z_offset = random.randint(min_z_offset, max_z_offset)
            prec_per_u = random.randint(min_prec_per_u, max_prec_per_u)
            prec_per_v = random.randint(min_prec_per_v, max_prec_per_v)
            color = color_array[random.randint(0, len(color_array) - 1)]
            (x, y, z) = createEllipse(width, height, depth, x_offset, y_offset, z_offset, prec_per_u, prec_per_v)
            ax.plot_surface(x, y, z, color=color)
        img_name = 'plot_' + str(img_n) + '.png'
        plt.savefig(save_path + img_name, bbox_inches="tight", pad_inches=0)
        gend_img = Image.open(save_path + img_name)
        crop_img = gend_img.crop((80, 40, img_width-200, img_height-150))
        crop_img.save(save_path + img_name, 'PNG')
        gend_img.close()
        plt.close(fig)


# Regular Images
generateImages(False)

# Images with Anomalies
generateImages(True)


"""
Gx, Gy = np.gradient(z) # gradients with respect to x and y
G = (Gx**2 + Gy**2)**.5  # gradient magnitude
N = G/G.max()
"""