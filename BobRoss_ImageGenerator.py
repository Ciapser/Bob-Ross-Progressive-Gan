import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2
from BobRoss_ProGan_Train import wasserstein_loss, MinibatchStdev, PixelNormalization, WeightedSum


def Load_generator(res, Fadein, path = None):
    """
    Loading generator
    
    Loading generator from projects folder, if there is no external path given, function
    will use parameters to find model in default path.

    Parameters:
        -res (int):
            Resolution of given model
            
        -Fadein (bool):
            Determines if loaded model will be Fadein variant or Straight variant
            
        -path (str):
            Path to model if it is present in other directory
    
    Returns:
        - loaded_generator (tf.keras.model):
            Trained tensorflow keras model ready to generate images in style of Bob ross
            in given resolution

    Notes:
        - There are custom function in model, be sure to import them from training script 
          or provide in some other way
    """
    #setting custom objects for model
    custom_objects = {
        'wasserstein_loss': wasserstein_loss,
        'MinibatchStdev': MinibatchStdev,
        'PixelNormalization': PixelNormalization,
        'WeightedSum': WeightedSum
                    }
    #Load model from other path if specified
    if path is not None:
        loaded_generator = tf.keras.load_model(path, custom_objects = custom_objects, compile = False)
        return loaded_generator
    
    #Load model, straight of fadein version
    if Fadein:
        loaded_generator = tf.keras.models.load_model("Models//generator_Fading_"+str(res)+".keras",custom_objects = custom_objects,compile = False)
    else:
        loaded_generator = tf.keras.models.load_model("Models//generator_"+str(res)+".keras",custom_objects = custom_objects, compile = False)
     
    return loaded_generator



def generate_images(generator, n_images, save_images = False):
    """
    Image generation
    
    Function generating images from given generator. Latent_dim should be the same
    as in the generator
    
    Parameters:
        -generator (tf.keras.model):
            Trained generator
            
        -n_images (int):
            Amount of images to generate
            
        -save_images (bool):
            Decides if saving images
  
    Returns:
        -Normalized (numpy array):
            Normalized batch of generated images in 0-255 uint8 format

    Notes:
        - No notes

    """
    #Checking input dimmension (lenght of latent_dim vector)
    latent_dim = int(generator.input.shape[1])
    #Generating random noise as input
    random_noise = np.random.randn(latent_dim*n_images)
    random_noise = np.reshape(random_noise, (n_images,latent_dim))
    
    #Generating images
    gen_images = generator.predict(random_noise)
    
    #Normalizing generated images (every image is normalized without splitting channels)
    normalized = np.array([(img - img.min()) / (img.max() - img.min()) for img in gen_images])
    #Changing format from 0-1 float to 0-255 uint8
    normalized = (normalized*255).astype(np.uint8)
    
    #Saving images if specified
    if save_images:
        np.save("Generated_images.npy", normalized)
    
    return normalized
    

def Plot_images(image_array, n_images = 25):
    """
    Plotting images
    
    Function plotting generated images
    
    Parameters:
        -image_array (numpy array):
            images to plot
            
        -n_images (int):
            Amount of images to plot
            
    Notes:
        - No notes

    """
    plt.figure()
    plt.suptitle("Gan generated images")
    indexes = [np.random.randint(0,len(image_array)) for i in range(n_images)]
    image_array = image_array[indexes]
    
    for i in range(n_images):
        plt.subplot(int(math.sqrt(n_images)),int(math.sqrt(n_images)),i+1)
        plt.axis("off")
        plt.imshow(image_array[i])
        plt.title("Index: "+str(indexes[i]))
    

def Compare_images(gen_images, real_images, n_images = 3):
    """
    Image comparision
    
    Function which takes random images from real dataset and compares
    them with generated ones
    
    Parameters:
        -gen_images (numpy array):
            Generated images dataset
            
        -real_images (numpy array):
            Real images dataset
            
        -n_images (int):
            Amount of comparisions of images, 1 means 1 real and 1 generated image

    Notes:
        - No notes

    """
    plt.figure()
    plt.suptitle("Generated and real image comparision")
    
    indexes = [np.random.randint(0,len(gen_images)) for i in range(n_images)]
    gen_images = gen_images[indexes]
    
    indexes = [np.random.randint(0,len(real_images)) for i in range(n_images)]
    real_images = real_images[indexes]
    
    res = gen_images.shape[1]
    real_images = [cv2.resize(img,(res, res),interpolation=cv2.INTER_NEAREST) for img in real_images]
    
    for i in range(n_images):
        plt.subplot(2,n_images,i+1)
        plt.imshow(real_images[i])
        plt.axis("off")
        plt.title("Real")
        
        plt.subplot(2,n_images,i+1+n_images)
        plt.imshow(gen_images[i])
        plt.axis("off")
        plt.title("Generated")
        
        

def main():
    #Loading generator from models folder
    generator = Load_generator(res = 256, Fadein = False)
    #Generating images and normalizing them to the 0-255 range in uint8 format
    gen_images = generate_images(generator = generator, n_images = 256, save_images = True)

    #Plotting generated images
    Plot_images(gen_images, 4)
    
if __name__ == "__main__":
    main()




"""
#Function to compare images with real ones, not necessary for generation and
needs real dataset to work so its commented

real_dataset = np.load("Bob_Ross_Filtered.npy")
Compare_images(gen_images,real_dataset)

"""

