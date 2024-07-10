#Main libraries
import sys
from NeuroUtils import ML_assets as ml
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#Additional libraries / shortcuts
from tensorflow.keras.initializers import HeNormal
from keras import backend
import math
from contextlib import redirect_stdout
import pandas as pd
import shutil

#1 Model architecture
####################################################################################################


#1.1
#Custom layers
class PixelNormalization(tf.keras.layers.Layer):
    """
    Pixelwise Feature Vector Normalization to use in GAN generator
    
    This function normalizes the feature vectors at each pixel location independently. 
    It computes the Euclidean norm of each feature vector and then divides the feature 
    vector by its norm, stabilizing training and improving gradient flow.
    
    Notes:
    - This technique is particularly useful in the context of GANs, helping to stabilize 
      training and improve the quality of generated images.
    - Ensure the input tensor has the appropriate shape and dtype for the operation.
    """
    #Initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

        
    #Perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs**2.0
        # calculate the mean pixel values
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized
    
    #Define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStdev(tf.keras.layers.Layer):
    """
    Minibatch Standard Deviation for GAN generator stabilization.
    
    This function computes the standard deviation of feature maps across a minibatch,
    and appends the computed standard deviation to each feature vector, helping to 
    introduce variability and stabilize GAN training.
    
    Notes:
    - This technique is particularly useful in GANs to encourage diversity in the 
      generated images and prevent mode collapse, because of diversity added as one of features
    - Ensure the input tensor has the appropriate shape and dtype for the operation. 
    """
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)


class WeightedSum(tf.keras.layers.Add):
    """
    Weighted Sum Layer for GAN resolution transition phase.

    This custom layer outputs a weighted sum of two input tensors. It is intended to use 
    in progressive GANs during the transition phase between different resolutions, preventing 
    resolution transition-induced model shock and allowing smooth blending of two sets of feature maps.

    
    Notes:
    - This layer only supports a weighted sum of exactly two input tensors.
    - The `alpha` attribute can be dynamically adjusted during training to control 
      the blending factor between the two inputs.

    """
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein loss for use in WGAN.
    
    This loss function computes the mean of the product of the true labels and the 
    predicted labels. It is designed to provide a continuous and differentiable 
    approximation to the Earth Mover's distance, which measures the distance between 
    probability distributions.
    
    Parameters:
        - y_true (tf.Tensor): 
            The true labels, which are typically -1 for real data and 1 for generated data.
        - y_pred (tf.Tensor): The predicted labels from the critic (discriminator).
    
    Returns:
        - tf.Tensor: The computed Wasserstein loss.
    
    Notes:
    - Ensure that `y_true` and `y_pred` have the same shape.
    """
    return backend.mean(y_true * y_pred)



#1.2
#Creating discriminator models
def add_discriminator_block(old_model,filters,new_learning_rate, n_input_layers=3):
    """
    Add a new block to the discriminator model for progressive GAN training.
    
    This function expands the discriminator model by adding a new block, effectively 
    doubling the input image resolution. It creates two models: one for the new 
    resolution and one that smoothly blends the new resolution with the previous 
    resolution using a weighted sum layer.
    
    Parameters:
        - old_model (tf.keras.Model): 
            The existing discriminator model to which a new block 
            will be added.
                                     
        - filters (int):              
            The number of filters for the convolutional layers in the new block.
        
        - new_learning_rate (float):  
            The learning rate for the Adam optimizer.
        
        - n_input_layers (int):       
            The number of layers to skip from the old model when 
            connecting the new block. Default is 3.
    
    Returns:
        A list containing two models:
          - `model1`: The expanded model for the new resolution (Straight model).
          - `model2`: The blended model for the transition phase (Fadein model).
    
    Notes:
        - The function assumes the input shape is square and doubles it for the new model.
        - The weighted sum layer (`WeightedSum`) is used to blend the old and new blocks 
          smoothly during the transition phase.
        - The models are compiled with the Wasserstein loss and the Adam optimizer with 
          specific hyperparameters.
    """
    #1 Shape and connectivity detection
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
    in_image = tf.keras.layers.Input(shape=input_shape)
    # check on top filters from model to connect smoothly with the same amount
    connection_filters = old_model.layers[n_input_layers].input_shape[-1]
    
    #2 Addition of new phase block
    # define new input processing layer
    d = tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_initializer=HeNormal())(in_image)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    
    # define new block
    d = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=HeNormal())(d)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.Conv2D(connection_filters, (3,3), padding='same', kernel_initializer=HeNormal())(d)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    d = tf.keras.layers.AveragePooling2D()(d)
    block_new = d
    
    #3 Remove n top layer
    # skip the input, 1x1 and activation for the old model (because of original 3 input layer removal)
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    
    #4 Straight model
    # define straight-through model
    model1 = tf.keras.models.Model(in_image, d)
    # compile model
    model1.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8))
    
    #5 Fadein model
    # downsample the new larger image
    downsample = tf.keras.layers.AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define fadein model
    model2 = tf.keras.models.Model(in_image, d)
    # compile model
    model2.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8))
    
    return [model1, model2]


def define_discriminator(n_blocks, filter_list, learning_rates, input_shape=(4,4,3)):
    """
    Define a list of discriminators for each resolution phase
    
    This function creates a discriminator model for a GAN that progressively grows 
    by adding new blocks. Each block increases the resolution of the input image 
    and the complexity of the model. The function returns a list of models for 
    different resolutions, with each resolution having a transition (Fadein) model 
    and a stable (straight) model.
    
    Parameters:   
        - n_blocks (int): 
            The number of resolution blocks to add to the discriminator.
    
        - filter_list (list of int): 
            A list of filter sizes for each block.
    
        - learning_rates (list of float):
            A list of learning rates for the Adam optimizer for each block.
    
        - input_shape (tuple of int): 
            The shape of the input images. Default is (4, 4, 3).
    
    Returns:
        - list: A list of model pairs (transition and stable) for each resolution.
    
    Notes:
        - Discriminator fadein and straighjt models are dependent of each other, 
          so it means they should be created by this function while training to provide weights exchange.
          While using discriminator it can be loaded just from file
    """
    model_list = []
    # base model input
    in_image = tf.keras.layers.Input(shape=input_shape)
    # conv 1x1
    d = tf.keras.layers.Conv2D(filter_list[0], (1,1), padding='same', kernel_initializer=HeNormal())(in_image)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = tf.keras.layers.Conv2D(filter_list[0], (3,3), padding='same', kernel_initializer=HeNormal())(d)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = tf.keras.layers.Conv2D(filter_list[0], (4,4), padding='same', kernel_initializer=HeNormal())(d)
    #d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = tf.keras.layers.GlobalAveragePooling2D()(d)
    out_class = tf.keras.layers.Dense(1)(d)
    # define model
    model = tf.keras.models.Model(in_image, out_class)
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[0], beta_1=0, beta_2=0.99, epsilon=10e-8))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model,filter_list[i],new_learning_rate = learning_rates[i])
        # store model
        model_list.append(models)
    return model_list


def add_generator_block(old_model, filters, new_learning_rate):
    """
    Add a new block to the generator model for progressive GAN training.
    
    This function expands the generator model by adding a new block, effectively 
    increasing the resolution of the output image. It creates two models: one for 
    the new resolution and one that smoothly blends the new resolution with the 
    previous resolution using a weighted sum layer.
    
    Parameters:
        - old_model (tf.keras.Model): 
            The existing generator model to which a new block will be added.
            
        - filters (int): 
            The number of filters for the convolutional layers in the new block.
            
        - new_learning_rate (float): 
            The learning rate for the Adam optimizer.
    
    Returns:
        A list containing two models:
          - `model1`: The expanded model for the new resolution.
          - `model2`: The blended model for the transition phase.
    
    Notes:
    - The function assumes the input shape is square and doubles it for the new model.
    - The weighted sum layer (`WeightedSum`) is used to blend the old and new blocks 
      smoothly during the transition phase.
    - The models are compiled with the Wasserstein loss and the Adam optimizer with 
      specific hyperparameters.
    """
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = tf.keras.layers.UpSampling2D()(block_end)
    g = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=HeNormal())(upsampling)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    g = tf.keras.layers.Conv2D(filters, (3,3), padding='same', kernel_initializer=HeNormal())(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = tf.keras.layers.Conv2D(3, (1,1), padding='same', kernel_initializer=HeNormal())(g)
    # define model
    model1 = tf.keras.models.Model(old_model.input, out_image)
    model1.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = tf.keras.models.Model(old_model.input, merged)
    model2.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]


def define_generator(latent_dim, n_blocks, filter_list, learning_rates, in_dim=4):
    """
    Define a progressive growing generator for a GAN.
    
    This function creates a generator model for a GAN that progressively grows
    by adding new blocks. Each block increases the resolution of the output image
    and the complexity of the model. The function returns a list of models for
    different resolutions, with each resolution having a transition (fadein) model 
    and a stable (straight) model.
    
    Parameters:
        - latent_dim (int): 
            The dimensionality of the latent input vector.
            
        -n_blocks (int): 
            The number of resolution blocks to add to the generator.
            
        - filter_list (list of int): 
            A list of filter sizes for each block.
            
        - learning_rates (list of float): 
            A list of learning rates for the Adam optimizer for each block.
            
        - in_dim (int): 
            The initial input dimension. Default is 4.
    
    Returns:
        - list: A list of model pairs (transition and stable) for each resolution.
    
    Notes:
        - Generator fadein and straighjt models are dependent of each other, 
          so it means they should be created by this function while training to provide weights exchange.
          While using generator it can be loaded just from file
    """
    model_list = []
    # base model latent input
    in_latent = tf.keras.layers.Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = tf.keras.layers.Dense(filter_list[0] * in_dim * in_dim, kernel_initializer=HeNormal())(in_latent)
    g = tf.keras.layers.Reshape((in_dim, in_dim, filter_list[0]))(g)
    # conv 4x4, input block
    g = tf.keras.layers.Conv2D(filter_list[0], (3,3), padding='same', kernel_initializer=HeNormal())(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = tf.keras.layers.Conv2D(filter_list[0], (3,3), padding='same', kernel_initializer=HeNormal())(g)
    g = PixelNormalization()(g)
    g = tf.keras.layers.LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = tf.keras.layers.Conv2D(3, (1,1), padding='same', kernel_initializer=HeNormal())(g)
    # define model
    model = tf.keras.models.Model(in_latent, out_image)
    model.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates[0], beta_1=0, beta_2=0.99, epsilon=10e-8))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model,filter_list[i],learning_rates[i])
        # store model
        model_list.append(models)
    return model_list



#2
#Data functions
def generate_real_samples(dataset, n_samples):
    """
    Generate real samples for training
    
    This function takes n amount of samples randomly and returns it
    for neural network to train on it. 
    
    Parameters:
        - dataset (numpy array): 
            numpy array containing samples. Samples should be stored along 0 axis,
            so the dataset[0] is first sample, dataset[1] is second etc...
        
        - n_samples (int): 
            number of samples to return from dataset
            
    Returns:
        - X (numpy array): Numpy array containing n amount of real samples from dataset
        - y (numpy array): Numpy array containing n amount of real samples labels, which are 1
    
    Notes:
        - Samples are selected randomly, so there should be provided enough training 
          steps to make sure they will come under law of big numbers
    """
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    """
    Generate latent point vector 
    
    This function creates latent vectors in lenght of latent dim, 
    and in amount of n_samples. Every row contains information for 
    one image to be created. Elements from latent dim vector are 
    generated randomly based on normal distribution
    
    Parameters:
        - latent_dim (int): 
            lenght of latent dim vector

        - n_samples (int): 
            number of vectors to generate
            
    Returns:
        - x_input (numpy array): 
            numpy array containing latent vector in every row, it has n_samples
            rows. Output of this function should be used as input for generating
            random images in generator

    Notes:
        - No notes
    """
    
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
    """
    Generate fake samples for training
    
    This function creates n amount of samples by generator and returns it.
    Then its rated by discriminator (critics)

    
    Parameters:
        - generator (tf.keras.model): 
            Generator trained to generate fake images of given dataset content
            
        - latent_dim (int): 
            latent_dim vector lenght, number of variables generated randomly 
            to create fake image from them
            
        - n_samples (int): 
            number of samples to generate
            
    Returns:
        - X (numpy array): Numpy array containing n amount of generated fake images
        - y (numpy array): Numpy array containing n amount of fake samples labels, which are -1
    
    Notes:
        - No notes
    """
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -np.ones((n_samples, 1))
    return X, y


def add_noise(images, noise_factor=0.1, range_min=-1.0, range_max=1.0):
    """
    Add Gaussian noise to the images.
    
    Parameters:
        - images (numpy array): 
            A batch of images selected to add gaussian noise.
        
        - noise_factor (float): 
            The standard deviation of the Gaussian noise.
        
        - range_min (float):
            Minimum value of the image range (buy default -1 for [-1:1] normalized images for gan)
        
        - range_max (float ): 
            Maximumvalue of the image range (buy default 1 for [-1:1] normalized images for gan)
    

    Returns:
        - Noisy images (numpy array)
        
    Notes:
        - No notes
    """
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy_images = images + noise
    # Clip the images to be in the specified range
    noisy_images = np.clip(noisy_images, range_min, range_max)
    return noisy_images   


def resize_scale_data(DataSet, Resolution, DataType = np.float32):
    """
    Resizing and scaling function
    
    This function resize the given DataSet to given resolution (square), and scales it
    to exact needs of this algorithm (for GAN network with -1 to 1 input for images)
    It uses nearest neighbour interpolation. It also uses DataType provided by user
    
    Parameters:
        - DataSet (numpy array): 
            Given DataSet to resize and rescale
        
        - Resolution (int): 
            Square resolution of resized dataset
        
        - DataType (dtype): 
            Datatype of resized dataet, it should be float dtype as it should
            support -1 to 1 range with enough resolution
                            
    Returns:
        - resized_data (numpy array)
        
    Notes:
        - No notes
    """
    resized_data = [cv2.resize(img,(Resolution, Resolution),interpolation=cv2.INTER_NEAREST) for img in DataSet]
    
    resized_data = np.array(resized_data, dtype = DataType)/255

    resized_data = (resized_data-0.5)*2
    
    return resized_data  


def Create_gif(input_res,output_res,fadein,fps):
    """
    Gif Creation
    
    This function creates gif based on dataset created during training loop
    Function is higlhy bounded with project and relies on its structure and 
    another function from NeuroUtils library.
    
    Parameters:
        - input_res (int): 
            resolution of saved dataset to load
        
        - output_res (int): 
            Output resolution of the gif to be reslcaled. Scaling is made by
            nearest neighbours so small pixelated image does not loose its nature
            after rescaling to bigger size
        
        - fadein (bool): 
            Determines if gif should be created of fadein or straight dataset 
            of choosen resolution
            
        - fps (int):
            Frames per second of created gif, higher fps is reccomended for longer
            training loops and lower fps for shorter training loops
                              
    Notes:
        - Function has no return but creates gif in specified gif location in project folder
    """
    if fadein:
        gif_array_file = os.path.join("Images","Gif_Data","Gif_"+str(input_res)+"x"+str(input_res)+"_Fadein.npy")
        save_gif_path = os.path.join("Images", "Gif" , "Gif_"+str(input_res)+"x"+str(input_res)+"_Fadein.gif") 
    else:
        gif_array_file = os.path.join("Images","Gif_Data","Gif_"+str(input_res)+"x"+str(input_res)+".npy")
        save_gif_path = os.path.join("Images","Gif" , "Gif_"+str(input_res)+"x"+str(input_res)+".gif") 
    
    gif_array = np.load(gif_array_file)
    
    gif_array = (gif_array - gif_array.min()) / (gif_array.max() - gif_array.min())
    gif_array =np.array(gif_array*255,dtype = np.uint8)
    
    ml.General.create_gif(gif_array, save_gif_path , output_res, output_res,fps = fps)



#3
#Training loop functions
def update_fadein(models, step, n_steps):
    """
    Update fadein layer impact

    Function is updating fadein layer alpha parameter lineary based on current step and given
    total amount of n_steps. Alpha parameter is responsible for smooth transition beetween 
    resolution phases and regulates % of output used from previous, only scaled resolution

    Parameters:
        - models (list of tf.keras.models): 
            List of fadein models, by default it should be [fadein_generator,fadein_discriminator]
        
        - step (int): 
            Current step of training loop
            
        - n_steps (int):
            Total amount of steps in training loop (for actual resolution phase)
            
    Notes:
        - Function has no return but is directly changing alphas in given models. Its working on
        given objects so it has no return and it manipulates given objects (model) inside the function
        with effect on the outside of the function.
    """
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


def Save_progress(generator, discriminator, fading, constant_noise, number_of_samples, resolution, current_epoch, d_loss, g_loss, model_folder_path = "Models", image_folder_path = "Images"):
    """
    Saving progress of trained models

    Function to put in training loop responsible to save training progress.
    It handles models saving, generating image for each training step (can be used for gif creation),
    and for saving data about trained progress and losses of the model.

    Parameters:
        - generator (tf.keras.model):
            Gan network generator
            
        -discriminator (tf.keras.model):
            Gan network discriminator
            
        -fading (bool):
            Informs function if it is inside fading phase or not
            
        - constant_noise (numpy array):
            constant input for the generator, thanks to the fact its constant and not random
            its possible to see smooth training process in generated pictures
            
        -number_of_samples (int):
            number of images generated by generator to save
            
        -resolution (int):
            Informs function of resolution in current phase of training
            
        -current_epoch (int):
            Informs function about current step (epoch) in training loop
            
        -d_loss (float):
            Discriminator loss to save in csv 
            
        -g_loss (float):
            Generator loss to save in csv
            
        -model_folder_path (str):
            path to the folder where models should be saved
            
        -image_folder_path (str):
            path to the main folder where generated images should be saved.
            It also creates other folders for the projects in that directory 
            if training is made from scratch
            
    Notes:
        - No notes
    """
    if fading:
        #Models_paths
        g_path = os.path.join(model_folder_path,"generator_Fading_"+str(resolution)+".keras")
        d_path = os.path.join(model_folder_path,"discriminator_Fading_"+str(resolution)+".keras")
        
        
    else:
        #Models_paths
        g_path = os.path.join(model_folder_path,"generator_"+str(resolution)+".keras")
        d_path = os.path.join(model_folder_path,"discriminator_"+str(resolution)+".keras")
        
    
    model_history_directory = "Model_history.csv"
    
    if resolution >4 or current_epoch>1:
        continue_training = True
    else:
        continue_training = False
        #Model folder
        if not os.path.isdir(model_folder_path):
            os.makedirs(model_folder_path)
        else:
            shutil.rmtree(model_folder_path)
            os.makedirs(model_folder_path)
            
        #Image folder
        if not os.path.isdir(image_folder_path):
            os.makedirs(image_folder_path)
        else:
            shutil.rmtree(image_folder_path)
            os.makedirs(image_folder_path)
            
        #Create additional folders
        os.makedirs(os.path.join(image_folder_path, "Gif_Data"))
        os.makedirs(os.path.join(image_folder_path, "Gif"))
        os.makedirs(os.path.join(image_folder_path, "Gen_Imgs"))
        os.makedirs(os.path.join(image_folder_path, "Gen_Real_Comparision"))

    #3
    #Saving history
    if continue_training:
        model_history = pd.read_csv(model_history_directory)
        
        next_index = len(model_history)  
        model_history.loc[next_index, 'epoch'] = current_epoch
        model_history.loc[next_index, 'resolution'] = resolution
        model_history.loc[next_index, 'fading'] = str(fading)
        model_history.loc[next_index, 'd_loss'] = d_loss
        model_history.loc[next_index, 'g_loss'] = g_loss
        
        
        model_history.to_csv(model_history_directory, index = False)
    else:
        c = ["epoch", "resolution", "fading", "d_loss", "g_loss"]
        model_history = pd.DataFrame(columns = c)
        
        next_index = len(model_history)  
        model_history.loc[next_index, 'epoch'] = int(current_epoch)
        model_history.loc[next_index, 'resolution'] = resolution
        model_history.loc[next_index, 'fading'] = str(fading)
        model_history.loc[next_index, 'd_loss'] = d_loss
        model_history.loc[next_index, 'g_loss'] = g_loss
        
        model_history.to_csv(model_history_directory, index = False)
        
        continue_training = True
    
    #Save imgs for gif
    with redirect_stdout(open(os.devnull, 'w')):
        
        gen = generator.predict(constant_noise)[0]
        gen = np.reshape(gen,(1,resolution,resolution,3))
        if fading:
            path = "Gif_"+str(resolution)+"x"+str(resolution)+"_Fadein.npy"
        else:
            path = "Gif_"+str(resolution)+"x"+str(resolution)+".npy"
            
        path = os.path.join("Images","Gif_Data",path)
        
        if os.path.isfile(path):
            loaded = np.load(path)
            gen = np.concatenate((loaded,gen),axis = 0)
        
        np.save(path,gen)
    
    
    #Save model weights to the folder
    print("\n")
    with redirect_stdout(open(os.devnull, 'w')):
        
        #Saving generator with weights and optimizer state
        generator.save(g_path)
        
        #Saving generator with weights and optimzier state
        discriminator.save(d_path)   

 
def gradient_penalty(discriminator, real_data, fake_data, gp_weight = 10):
    """
    Gradient loss penalty
    
    Responsible for enforcing the Lipschitz constraint inside the discriminator, 
    which keeps its gradients bounded, typically around a norm of 1. This ensures 
    that the theoretical properties of the Wasserstein distance in a WGAN are satisfied.
    In practice, it helps stabilize the training process and improves the quality of 
    the outputs generated by the generator.
        
    Parameters:
        - discriminator (tf.keras.model):
            discriminator to which gradient penalty should be applied
            
        -real_data (numpy array):
            batch of real samples provided in the training step
            
        -fake_data (numpy array):
            batch of fake samples generated by generator in the training step
            
        -gp_weight (int):
            gradient penalty strenght multiplier, by default 10
    
    Returns:
        - gradient_penalty (float): 
            gradient penalty to apply to discriminator
            
    Notes:
        - No notes
    """
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)  # Ensure alpha matches real_data dimensions
    interpolated = alpha * real_data + (1 - alpha) * fake_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    gradients = tape.gradient(pred, [interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
    
    return gp_weight * gradient_penalty


def train_epochs(g_model, d_model, dataset, n_epochs, n_batch,latent_dim,constant_noise,number_of_samples, fadein=False,starting_epoch = 0,n_discriminator = 1):
    """
    Single phase training function
    
    Function executing training given models for resolution phase. It is single phase
    and can be either straight or fadein phase. it takes given parameters and data to
    train the models. It also saves progress after finishing each training epoch using
    "Save_progress" function, and creates gif after finishing phase by function "Create_gif"
  
    Parameters:
        -g_model (tf.keras.model):
            generator of gan model
            
        -d_model (tf.keras.model):
            discriminator of gan model 
            
        -n_epochs (int):
            Total amount of epochs for this training phase
        
        -n_batch (int):
            batch size for this training phase
            
        - constant_noise (numpy array):
            constant input for the generator, thanks to the fact its constant and not random
            its possible to see smooth training process in generated pictures in "Saving_progress" function
            
        -number_of_samples (int):
            number of images to save in "Saving progress" function
        
        -fadein (bool):
            Informs the function if its fadein phase or not
            
        -starting_function (int):
            By default 0, it informs function from which epoch of training it should start working
            If training was interrupted and continued, function will renew in exact epoch it stopped
            
        -n_discriminator (int):
            By default 1, it determines how many times discriminator is trained per
            single training phase of generator
                 
    Notes:
        - No notes
    """ 
    # calculate the number of batches per training epoch
    res = int(dataset.shape[1])
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    if fadein:
        Type = 'Fadein'
    else:
        Type = "Straight"
    description = "Res: "+str(res)+"x"+str(res)+"  "+Type
    #########
    for epoch in range(starting_epoch, n_epochs):
        print("\n\nEpoch",epoch+1,"/",n_epochs)
        for i in tqdm(range(bat_per_epo),desc = description):
            with redirect_stdout(open(os.devnull, 'w')):
                # Update alpha for all WeightedSum layers when fading in new blocks
                if fadein:
                    update_fadein([g_model, d_model], epoch * bat_per_epo + i, n_steps)
                    
                # Train the discriminator
                for _ in range(n_discriminator):
                    # Prepare real and fake samples
                    X_real, y_real = generate_real_samples(dataset, half_batch)
                    X_real = add_noise(X_real, noise_factor = 0.02)
                    
                    noise = tf.random.normal([half_batch, latent_dim])
                    X_fake = g_model(noise, training=True)
                    lambda_gp = 10
                    with tf.GradientTape() as disc_tape:
                        real_output = d_model(X_real, training=True)
                        fake_output = d_model(X_fake, training=True)
                        gp = gradient_penalty(d_model, X_real, X_fake, lambda_gp)
                        disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp
                
                    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)
                    d_model.optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
                
                # Train the generator
                noise = tf.random.normal([n_batch, latent_dim])
                with tf.GradientTape() as gen_tape:
                    fake_data = g_model(noise, training=True)
                    fake_output = d_model(fake_data, training=False)
                    gen_loss = -tf.reduce_mean(fake_output)
                
                gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
                g_model.optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
        
        sys.stdout.write('Epoch: %d, d_loss=%.3f, g_loss=%.3f' % ((epoch+1, disc_loss, gen_loss)))
        Save_progress(generator = g_model ,
                      discriminator = d_model,
                      fading = fadein,
                      constant_noise = constant_noise,
                      number_of_samples = number_of_samples,
                      resolution = res,
                      current_epoch = epoch+1,
                      d_loss = float(disc_loss),
                      g_loss = float(gen_loss)
                      )
        
    Create_gif(input_res = res, output_res = 512, fadein = fadein, fps = 10)
        

def summarize_performance(status, g_model, dataset, constant_noise, gen_samples=16, comp_samples = 5):
    """
    End of phase model summary of performance
    
    It generates images from model in finished phase (fadein or straight), and saves them 
    to folder. It creates comparision of generated images with real images downsampled to 
    current resolution of training phase
  
    Parameters:
        -status (str):
            ("Fadein" / "Straight") status of phase used in creation of title in plots 
            
        -g_model (tf.keras.model):
            generator of gan model
            
        -dataset (numpy array):
            full dataset used in training phase. Used to pick random samples for comparision
            
        -gen_samples (int):
            Amount of samples to put in the plot of generated images only
            
        -comp_samples (int):
            Amount of fake samples to put in comparision plot, the same amount 
            of real samples will be put in the folder on the opposite side
                    
    Notes:
        - Too big amount of samples can make plots unreadable, and make judge of 
          generated image quality impossible.
    """ 
    #1
    #Only generated images saved
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    gen = g_model.predict(constant_noise)[0:gen_samples]
    # normalize pixel values to the range [0,1]
    gen = (gen - gen.min()) / (gen.max() - gen.min())
    if np.isnan(gen).any():
        print("Found nan values in generated images, replacing to 0...")
        print("This often indicates error of exploding gradients, check your network to eliminate errors")
        gen[np.isnan(gen)] = 0
    #plot generated images
    square = int(math.sqrt(gen_samples))
    title = "Res "+str(gen_shape[1]) + "x" + str(gen_shape[2]) +" " +status
    plt.suptitle(title)
    for i in range(gen_samples):
        plt.subplot(square, square, 1 + i)
        plt.axis('off')
        plt.imshow(gen[i])
    # save plot to file
    filename = 'Images\\Gen_Imgs\\plot_%s.png' % (name)
    plt.savefig(filename)
    plt.close()
    
    #2
    #Generated and real imgs comparision
    res = gen_shape[1]

    gen = g_model.predict(constant_noise)[0:comp_samples]
    gen = (gen - gen.min()) / (gen.max() - gen.min())
    
    indexes = [np.random.randint(0,len(dataset)) for i in range(comp_samples)]
    dataset = dataset[indexes]
    dataset = resize_scale_data(DataSet = dataset, Resolution = res, DataType = np.float32)
    dataset = (dataset+1) / 2
    
    
    #Comparing images
    plt.figure()
    title = "Res: "+str(res)+"x"+str(res)+"  Status: "+status
    plt.suptitle(title)
    for i in range(comp_samples):
        plt.subplot(2,comp_samples,i+1)
        plt.imshow(dataset[i])
        plt.axis("off")
        plt.title("Real")
        
        plt.subplot(2,comp_samples,i+1+comp_samples)
        plt.imshow(gen[i])
        plt.axis("off")
        plt.title("Fake")
        
    #Saving plot
    filename = 'Images\\Gen_Real_Comparision\\Comparision_plot_%s.png' % (name)
    plt.savefig(filename)
    plt.close()
    
    
def train(g_models, d_models, dataset, latent_dim, epoch_straight, epoch_fadein, n_batch, start_block, start_epoch, train_Fadein,constant_noise, n_discriminator = 1):
    """
    Full GAN train function 
    
    Full GAN train function through all phases. It supervises training process

  
    Parameters:
        - g_models (list of lists of tf.keras.models):
            list containing list of straight and fadein generators for each phase where 
            phase 1: [straight,straight] (as it does not have fadein)
            phase 2: [straight,fadein]
            phase3: [straight,fadein]
            ...
            
        - d_models (list of lists of tf.keras.models):
            list containing list of straight and fadein discriminators for each phase where 
            phase 1: [straight,straight] (as it does not have fadein)
            phase 2: [straight,fadein]
            phase3: [straight,fadein]
            ...
            
        - latent_dim (int):
            latent_dim vector lenght
            
        - epoch_straight (list of int):
            list of amount of epochs to train model through each straight resolution phase
            
        - epoch_fadein (list of int):
            list of amount of epochs to train model through each fadein resolution phase
            
        - n_batch (list of int):
            list of batch sizes to use in training resolution phase
            
        - start_block (int):
            starting block (phase), if its zero model is starting from first phase
            
        - start_epoch (int):
            starting epoch for training loop
            
        - train_Fadein (bool):
            Informs model if it should train fadein phase for resolution 
            If it has been already finished before, it is set to False as it wont train Fadein
            
        - constant_noise (numpy array):
            constant input for the generator, thanks to the fact its constant and not random
            its possible to see smooth training process in generated pictures in "Saving_progress" function
            
        -n_discriminator (int):
            By default 1, it determines how many times discriminator is trained per
            single training phase of generator
                    
    Notes:
        - In this project it is handled that models are bounded with each other,
          however if this function is used standalone, make sure that provided models
          are exchanging its weights between fadein/straight phase, and between 
          resolution phases at all. You can make it by copying equivalent weights
          between the models or creating composite models.
    """ 
    if start_block ==0:
        # fit the baseline model
        g_normal, d_normal = g_models[0][0], d_models[0][0]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = resize_scale_data(DataSet = dataset, Resolution = int(gen_shape[1]), DataType = np.float32)
        print('Scaled Data', scaled_data.shape)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, scaled_data, epoch_straight[0], n_batch[0],latent_dim, constant_noise, n_batch[0], fadein = False, starting_epoch = start_epoch, n_discriminator = n_discriminator )
        summarize_performance('Straight',g_normal, dataset, constant_noise)
        
        start_block+=1
        train_Fadein = True
        start_epoch = 0
    
    # process each level of growth
    for i in range(start_block, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = resize_scale_data(DataSet = dataset, Resolution = int(gen_shape[1]), DataType = np.float32)
        print('Scaled Data', scaled_data.shape)
        
        if train_Fadein:
            # train fade-in models for next level of growth
            train_epochs(g_fadein, d_fadein, scaled_data, epoch_fadein[i], n_batch[i],latent_dim, constant_noise, n_batch[0], True, starting_epoch = start_epoch)
            summarize_performance('Faded',g_fadein, dataset, constant_noise)
            start_epoch = 0
            
            
            # train normal or straight-through models
            train_epochs(g_normal, d_normal, scaled_data, epoch_straight[i], n_batch[i],latent_dim, constant_noise, n_batch[0], False, starting_epoch = start_epoch)
            summarize_performance('Straight',g_normal, dataset, constant_noise)
            start_epoch = 0
            
        else:
            # train normal or straight-through models
            train_epochs(g_normal, d_normal, scaled_data, epoch_straight[i], n_batch[i],latent_dim, constant_noise, n_batch[0], False, starting_epoch = start_epoch)
            summarize_performance('Straight',g_normal, dataset, constant_noise)
            start_epoch = 0
            train_Fadein = True
            

def Continue_training(d_models, g_models,learning_rates, model_history_path):
    """
    Continuity of training function
    
    Checks model history file and based on information from it, recreates the state 
    of training where it has been stopped. 

  
    Parameters:
        - g_models (list of lists of tf.keras.models):
            list containing list of straight and fadein generators for each phase where 
            phase 1: [straight,straight] (as it does not have fadein)
            phase 2: [straight,fadein]
            phase3: [straight,fadein]
            ...
            
        - d_models (list of lists of tf.keras.models):
            list containing list of straight and fadein discriminators for each phase where 
            phase 1: [straight,straight] (as it does not have fadein)
            phase 2: [straight,fadein]
            phase3: [straight,fadein]
            ...
            
        - learning_rates (list of float):
            learning rates used in each resolution phase
            
        - model_history path (str):
            directory to the model_history csv file containing data about the training
            loss, current epoch, current phase

    Returns:
        - d_models (list of lists of tf.keras.models):
            modified input d_models, they are updated by trained weights and optimizer state
            
        - g_models (list of lists of tf.keras.models):
            modified input g_models, they are updated by trained weights and optimizer state
            
        -start_block (int):
            Phase from which training should be renewed
            
        -start_epoch (int):
            epoch from which training should be renewed
            
        -train_Fadein (bool):
            determines if training should be renewed from fadein or straight phase of resolution
        
    Notes:
        - This function takes and returns models to ensure they are composite to each other after
          parameter injection, if this function would only load models they would not be bounded.

    """ 
    #Setting custom objects
    custom = {
            'wasserstein_loss': wasserstein_loss,
            'MinibatchStdev': MinibatchStdev,
            'PixelNormalization': PixelNormalization,
            'WeightedSum': WeightedSum
            }
    
    
    
    if os.path.exists(model_history_path):
        print("Model history exists, renewing training...")
        train_from_scratch = False
    else:
        print("Model history does not exists, training from scratch...")
        train_from_scratch = True
      
    if train_from_scratch:
        start_block = 0
        start_epoch = 0
        train_Fadein = False
        
        print("No model history found, training from scratch...")
        return d_models,g_models,start_block,start_epoch,train_Fadein
        
        
    else:
        print("Found model history file, loading saved results...")
        m_history = pd.read_csv(model_history_path)
        record = m_history.iloc[[-1]].reset_index(drop = True)
        
        start_epoch = int(record["epoch"][0])
        
        res = int(record["resolution"][0])
        start_block = int(round(math.log(res/4,2)))
        train_Fadein = record["fading"][0]
    
    #############################################

        for i in range(0,start_block+1):
            res = 4*(2**i)

            if i ==0:
                print("\nLoading ",res,"x",res,"models")
                
                print("Loading discriminator")
                #Loading optimizer
                d_path = str("Models//discriminator_"+str(res)+".keras")
                d_model = tf.keras.models.load_model(d_path, custom_objects = custom)
                d_opt = d_model.optimizer
                d_weights = d_model.get_weights()
                
                #Compiling model to retrieve optimizer
                d_models[i][0].compile(loss = wasserstein_loss, optimizer = d_opt)
                #Loading model
                d_models[i][0].set_weights(d_weights)
                #Coping model to list as its first resolution, just for keeping size of list
                d_models[i][1] = d_models[i][0]
                
                print("Loading generator")
                #Loading optimizer
                g_path = str("Models//generator_"+str(res)+".keras")
                g_model = tf.keras.models.load_model(g_path, custom_objects = custom)
                g_opt = g_model.optimizer
                g_weights = g_model.get_weights()
                
                #Compiling model to retrieve optimizer
                g_models[i][0].compile(loss = wasserstein_loss, optimizer = g_opt)
                #Loading model
                g_models[i][0].set_weights(g_weights)
                #Coping model to list as its first resolution, just for keeping size of list
                g_models[i][1] = g_models[i][0]
                
                
            if i>0:
                print("\nLoading ",res,"x",res,"models")
                print("Loading Fading...")
                print("Loading discriminator")
                #Loading optimizer
                d_path = str("Models//discriminator_Fading_"+str(res)+".keras")
                d_model = tf.keras.models.load_model(d_path, custom_objects = custom)
                d_opt = d_model.optimizer
                d_weights = d_model.get_weights()
                
                #Compiling model to retrieve optimizer
                d_models[i][1].compile(loss = wasserstein_loss, optimizer = d_opt)
                #Loading model
                d_models[i][1].set_weights(d_weights)
                
                
                print("Loading generator")
                #Loading optimizer
                g_path = str("Models//generator_Fading_"+str(res)+".keras")
                g_model = tf.keras.models.load_model(g_path, custom_objects = custom)
                g_opt = g_model.optimizer
                g_weights = g_model.get_weights()
                
                #Compiling model to retrieve optimizer
                g_models[i][1].compile(loss = wasserstein_loss, optimizer = g_opt)
                #Loading model
                g_models[i][1].set_weights(g_weights)


                
                if not (i == start_block and train_Fadein):
                    print("Loading straight...")
                    print("Loading discriminator")
                    #Loading optimizer
                    d_path = str("Models//discriminator_"+str(res)+".keras")
                    d_model = tf.keras.models.load_model(d_path, custom_objects = custom)
                    d_opt = d_model.optimizer
                    d_weights = d_model.get_weights()

                    #Compiling model to retrieve optimizer
                    d_models[i][0].compile(loss = wasserstein_loss, optimizer = d_opt)
                    #Loading model
                    d_models[i][0].set_weights(d_weights)

                    
                    print("Loading generator")
                    #Loading optimizer
                    g_path = str("Models//generator_"+str(res)+".keras")
                    g_model = tf.keras.models.load_model(g_path, custom_objects = custom)
                    g_opt = g_model.optimizer
                    g_weights = g_model.get_weights()
                    
                    #Compiling model to retrieve optimizer
                    g_models[i][0].compile(loss = wasserstein_loss, optimizer = g_opt)
                    #Loading model
                    g_models[i][0].set_weights(g_weights)


                

        return d_models,g_models,start_block,start_epoch,train_Fadein



#4
#Additional functions involved in project creation but not used directly in final version
def check_compositivity(fadein_model,straight_model):
    """
    Compositivity check 
    
    Checks model history file and based on information from it, recreates the state 
    of training where it has been stopped. 

  
    Parameters:
        - fadein_model (tf.keras.model):
            fadein model, or any composite model to check
            
        - straight_model (tf.keras.model):
            straight model, or any composite model to check
            
    Notes:
        - This function affects the models as it overwrites ones weights with random noise
          If models are composite, second one will be overwrited too. Models, after ensuring 
          that they are bounded, should be build and compiled again with the same steps and 
          not putted through this function.
    """ 
    
    #Loading models weights and checking if they are the same (should be)
    weights_straight = straight_model.layers[1].get_weights()
    weights_fadein = fadein_model.layers[1].get_weights()
    weight_test1 = np.array_equal(weights_straight[0],weights_fadein[0])
    if weight_test1:
        print("Models have identical weights before manipulating weights")
    else:
        print("ERROR: Models do not have identical weights before manipulating weights")
    
    
    changed_weights = np.random.randn(weights_fadein[0].shape[0]*weights_fadein[0].shape[1])
    changed_weights = np.reshape(changed_weights, newshape = weights_straight[0].shape)
    
    changed_fadein_weights = weights_fadein.copy()
    changed_fadein_weights[0] = changed_weights
    
    #changing straight weights 
    fadein_model.layers[1].set_weights(changed_fadein_weights)
    
    
    weights_straight2 = straight_model.layers[1].get_weights()
    weights_fadein2 = fadein_model.layers[1].get_weights()
    weight_test2 = np.array_equal(weights_straight2[0],weights_fadein2[0])
    if weight_test2:
        print("Models have identical weights after manipulating weights")
    else:
        print("ERROR: Models do not have identical weights after manipulating weights")
        
    if weight_test1 and weight_test2:
        print("Models are composite, they are connected and change in one does affect another")
    else:
        print("Models are not composite, they are not connected and change in one does not affect another")
        

def Plot_model_history(model_history_path):
    """
    Plots history
    
    Function plotting history of model training
    
    Parameters:
        - model_history_path (str):
            model history directory
    """
    m_history = pd.read_csv(model_history_path)
    
    plt.figure()
    plt.plot(m_history["d_loss"],label = "d_loss",c = "red")
    plt.plot(m_history["g_loss"],label = "g_loss",c = "green")
    plt.legend()




#5
#Main function
def main(gpu_memory_limit, model_history_path, dataset_path, phases, latent_dim, f_list_discriminator, f_list_generator, n_epochs, n_batch, l_rates, n_discriminator = 1 ):
    """
    Main function
    
    Main function merging all functions to perform complete training of BobRoss_ProGan
  
    Parameters:
        - gpu_memory_limit (int):
            gpu memory limit in bytes, if you do not have enough Vram on gpu you may need
            switch to CPU or reduce batch_size
            
        - model_history_path (str):
            model history directory
            
        - dataset_path (str):
            dataset directory
            
        - phases (int):
            how many phases should model train for, 1 stands for 4x4 output, 2 for 8x8 etc...
            
        - latent_dim (int):
            lenght of latent_dim vector
            
        - f_list_discriminator (list of int):
            amounts of filters for discriminator each growth phase
            
        - f_list_generator (list of int):
            amounts of filters for generator each growth phase   
            
        - n_epochs (list of int):
            amount of epochs to train for each growth phase
            
        - n_batch (list of int):
            batch sizes for training loops in each growth phase
            
        - l_rates (list of float):
            learning rates for each growth phase models
            
        - n_discriminator (int):
            By default 1, it determines how many times discriminator is trained per
            single training phase of generator
            
    Notes:
        - No Notes

    """ 
    #1
    #Hardware check and configuration
    
    #Printing tensorflow version installed
    print(f"TensorFlow version: {tf.__version__}")
    
    #Checking if gpu is present and detected
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
    else:
        print("No GPU detected")
    
    #Trying to set memory limit for gpu
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])
        except RuntimeError as e:
            print(e)
            
    #2
    #Training preparation

    #Create constant_noise to compare generated images
    if not os.path.isfile("constant_noise.npy"):
        c_n = np.random.normal(0, 1, (n_batch[0], latent_dim))
        np.save("constant_noise.npy",c_n)
        print("Constant noise created and saved into project folder")
    else:
        c_n = np.load("constant_noise.npy")
        print("Detected and loaded existing constant noise")

    # define discriminators
    d_models = define_discriminator(n_blocks = phases,filter_list = f_list_discriminator, learning_rates = l_rates)

    # define generators
    g_models = define_generator(latent_dim = latent_dim, n_blocks = phases,filter_list = f_list_generator, learning_rates = l_rates)

    # load image data
    Dataset = np.load(dataset_path)
    print('Loaded', Dataset.shape)

    #Check if there was training done before. If so, training progress is loaded 
    d_models,g_models,start_block,start_epoch,train_Fadein = Continue_training(d_models = d_models,
                                                                               g_models = g_models,
                                                                               learning_rates = l_rates,
                                                                               model_history_path = model_history_path
                                                                               )    
    #Checking if there is gpu (if it is present use it instead of cpu)
    device = "CPU:0"
    if gpus:
        device = "GPU:0"
        
    #3
    # train model   
    with tf.device(device):
        train(g_models = g_models,
              d_models = d_models,
              dataset = Dataset,
              latent_dim = latent_dim,
              epoch_straight = n_epochs,
              epoch_fadein = [e//2 for e in n_epochs],
              n_batch = n_batch,
              start_block = start_block,
              start_epoch = start_epoch,
              train_Fadein = train_Fadein,
              constant_noise = c_n,
              n_discriminator = n_discriminator
              )

    
if __name__ == "__main__":
    
    """
    #Hyperparameters tested working well
    latent_dim = 512
    n_discriminator = 1
    f_list = [256,256,256,256,256]
    f_list_generator = [512,512,512,512,512]
    n_epochs = [200,200,200,200,200]

    n_batch = [256, 256, 256, 16,16]
    l_rates = [1e-3,1e-4,1e-4,1e-4,1e-4]


    """
    #Phases for output resolution
    #1 = 4x4
    #2 = 8x8
    #3 = 16x16
    #4 = 32x32
    #5 = 64x64
    #6 = 128x128
    #7 = 256x256
    #8 = 512x512
    #9 = 1024x1204 
    
    main(gpu_memory_limit =                     11776,
         model_history_path =                   "Model_history.csv",
         dataset_path =                         "Bob_Ross_Filtered.npy",
         phases =                               8,
         latent_dim =                           512,
         f_list_discriminator =                 [256,256,256,256,256,128,64,32],
         f_list_generator =                     [512,512,512,512,512,256,128,64],
         n_epochs =                             [200,200,200,200,200,200,200,200],
         n_batch =                              [256, 128, 64, 16, 16, 16,6,3],
         l_rates =                              [1e-3,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4],
         n_discriminator =                      1
         )

















