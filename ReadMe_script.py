from NeuroUtils import ML_assets as ml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
from BobRoss_ImageGenerator import Load_generator
#############################################################################
#1
#Nice image for thumbnail

gen_imgs = np.load("Generated_images.npy")
idx = 92
img = cv2.cvtColor(gen_imgs[idx], cv2.COLOR_RGB2BGR)
cv2.imwrite("Github_Readme_files\\Thumbnail_img.png", img )
#############################################################################
#2
#Image Checkboard

#Random idx generation without repetitions
idx_1 = np.random.choice(range(len(gen_imgs)), 18, replace=False)

#Idx splitting
idx_2 = idx_1[0:9]
idx_1 = idx_1[9:18]

#Creating checkboards from images
checkboard_1 = ml.General.create_image_grid(img_array = gen_imgs[idx_1], size = 3, RGB = True)
checkboard_2 = ml.General.create_image_grid(img_array = gen_imgs[idx_2], size = 3, RGB = True)

#Showing checkboards
plt.figure()
plt.imshow(checkboard_1)
plt.figure()
plt.imshow(checkboard_2)

#Saving checkboards with cv2 bgr format
cv2.imwrite("Github_Readme_files\\Generated_imgs_checkboard_1.png",cv2.cvtColor(checkboard_1, cv2.COLOR_RGB2BGR))
cv2.imwrite("Github_Readme_files\\Generated_imgs_checkboard_2.png",cv2.cvtColor(checkboard_2, cv2.COLOR_RGB2BGR))
#############################################################################
#3
#Gif of all training

#Specifying gif data folder elements
gif_dirs = os.listdir("Images\\Gif_Data")
# Custom sorting key
def sort_key(filename):
    # Extract number from the filename
    number = int(re.search(r'\d+', filename).group())
    # Check if 'Fadein' is in the filename
    fadein = 'Fadein' in filename
    # Return a tuple for sorting
    return (number, not fadein)

#Sort paths by givn key
gif_dirs = sorted(gif_dirs, key = sort_key)

#Load arrays
gif_arrays = [np.load(os.path.join("Images", "Gif_Data", path)) for path in gif_dirs]
#Setting image resolution
r_size = 256
#Creating empty array to store resized arrays of gifs
full_gif = np.empty((0,r_size,r_size,3),dtype = np.uint8)

#Resizing and normalizing images
for array in gif_arrays:
    normalized = np.array([(img - img.min()) / (img.max() - img.min()) for img in array])
    
    resized_array = [cv2.resize(iteration,(r_size,r_size), interpolation = cv2.INTER_NEAREST_EXACT) for iteration in normalized]
    resized_array = np.array(resized_array)
    resized_array = (resized_array*255).astype(np.uint8)
    full_gif = np.concatenate((full_gif,resized_array),axis = 0)

#gif creation 
ml.General.create_gif(full_gif, "Github_Readme_files\\Full_train_gif.gif", gif_height = 256, gif_width = 256,fps = 10)
#############################################################################
#4
#Image interpolation

#Loading model to use in function
generator = Load_generator(256, False)

#Loading interpolated images and slider to create a plot
interpolated_images, slider = ml.General.Image_interpolation(generator = generator,
                                                            n_variations = 5,
                                                            steps_to_variation = 225 ,
                                                            is_grayscale = False,
                                                            create_gif = True,
                                                            gif_path = "Github_Readme_files\\Model_interpolation.gif", 
                                                            gif_scale = 1,
                                                            gif_fps = 15
                                                            )



