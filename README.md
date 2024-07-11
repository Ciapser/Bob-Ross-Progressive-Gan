# Bob Ross Progressive GAN
![alt text](https://github.com/Ciapser/Bob-Ross-Progressive-Gan/blob/master/Github_Readme_files/Thumbnail_img.png?raw=true)

This project aims to allow computer have their own joy of painting, and allow them to paint their "happy little tree".

It is based on Progressive Generative Adversarial Networks, which is variant of GAN allowing for creating high resolution images. 

Dataset of 5000 images in the style of Bob Ross was used to train the model. 

### Project contains scripts for:
- Model training                            (BobRoss_ProGan_Train.py)
- Reneweing training if it crashed          (Run_Reset.py)
- creating assets for ReadMe presentation   (ReadMe_script.py)
- Generating images                         (BobRoss_ImageGenerator.py)

### Due to the file size, project does not contains:
- Training dataset
- Every phase model (only last generator and discriminator are present)

## Explanation of concept
### Gan - Generative adversarial networks
Gans contains 2 network which are competing with each other. One of them is generator and the other one is discriminator. Generator is responsible for creating imagesout of noise, and discriminator is responsible for classificating the image to the fake or real label. 

Discriminator is learned based on the real dataset and generator is learned based on discriminator error - if discriminator classified fake image as real, generator enforces pattern behind creation of that image.
### ProgGan - Progressive Generative adversarial networks
Basics Gans are not capable of creating images of high resolution. 
Thats why Progressive Gan was intruduced. Instead of trying to create big lets say 256x256 image from the beginning, models are competing at smaller resolution: 4x4, then jumping to 8x8, 16x16 etc. 

Thanks to that technique its possible for generator to learn how to create good quality high resolution image.

Techniques which are used in ProGans for training:
- Progressive growing,
- WGAN loss with gradient penalty,
- Fadein layers,
- Std BatchNormalization,
- Pixel Normalization,
- Equalized learning rate,
- specific weights initialization

You can find more detailed information and explanations in the Credits and good articles section, which I highly encourage to read if you are interested in the topic.


## Usage
### Training model
To train the network you should put dataset of RGB images in the numpy format, in the main project folder. By default it is named "Bob_Ross_Filtered.npy" , but you can change it in the script. Then run BobRoss_ProGan_Train.py. Training can be interrupted and renewed without losing progress [(1) Look into  Issues!]

    Hyperparameters can be adjusted in the main function
### Generating images
To generate images simply run BobRoss_ImageGenerator.py and images will be saved in the numpy format in project directory. To use this script you will need trained model which is available to download.

If you want to manipulate the images or view them you can use
ReadMe_script.py, however this is more customized to the needs of
this ReadMe file. But it contains some functions ready to view
generated images. 
    
    !Beware that by running these scripts you will overwrite files in
    "Github_Readme_files" folder and "Generated_images.npy" in project
    folder!
    
    
## Results

### Generator outputs:
There are some of randomly generated images from the generator
![alt text](https://github.com/Ciapser/Bob-Ross-Progressive-Gan/blob/master/Github_Readme_files/Generated_imgs_checkboard_1.png?raw=true)


![alt text](https://github.com/Ciapser/Bob-Ross-Progressive-Gan/blob/master/Github_Readme_files/Generated_imgs_checkboard_2.png?raw=true)

As you can see there are visible some artifacts or abstract views on images, however other images represents quite good quality and are very pleasent to the eye. 
Even better results will be probably obtained if bigger dataset would be used.

### Outputs interpolation
Here is gif showing smooth transition between few generated images.
Here is my interpretation of landscapes over time:          

Bayou sunset --> Mountain lake --> Bigger mountain lake --> 
High mountains --> Waves --> Snow plains --> Bayou sunset

![alt text](https://github.com/Ciapser/Bob-Ross-Progressive-Gan/blob/master/Github_Readme_files/Model_interpolation.gif?raw=true)

### Generator training:
Here is representation of generator outputs over training in every resolution and phase. Output was based on the constant, not changed input.

Starting resolution is 4x4 and it goes up to 256x256 through total amount of 2k epochs.
![alt text](https://github.com/Ciapser/Bob-Ross-Progressive-Gan/blob/master/Github_Readme_files/Full_train_gif.gif?raw=true)


## Possible future goals:
 - Simple increase in the resolution of image (more training)
 - Preparing bigger dataset to train model on
 - Allow for more custom output shape ratio, fe. 3:5, not only squares


## Credits and good articles:
- Very good explanation of the original paper introducing this technique:  

    https://youtu.be/lhs78if-E7E
- Good explanation with code, examples and pictures:

    https://blog.paperspace.com/progan/
- Example of code, with emphasis on model creation and composite aspect:

    https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/