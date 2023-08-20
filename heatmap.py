import numpy as np
import numpy as np
import matplotlib.cm as cm
# from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
# from tensorflow.keras.preprocessing import image
from keras.utils import load_img,img_to_array
import cv2

heatmap_model=tf.keras.models.load_model("./Models/4 way classifier model- stacked images.h5")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        preds= preds[0]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel,last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def heatmapImage(image_stacked_arr, heatmap):
    # Load the original image
    # path="../Images/stacked.png"
    # img = tf.keras.preprocessing.image.load_img(path)
    # img = tf.keras.preprocessing.image.img_to_array(img)

    # Load the original stacked image array
    img = image_stacked_arr

    # constant
    alpha = 0.4
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)
    return np.asarray(superimposed_img)

    # Display Grad CAM
    # display(Image(cam_path))

def heatMapArr(stacked_image_array: np.ndarray) -> np.ndarray:
    path="./Images/stacked.png"
    img = load_img(path,target_size=(256,256))
    # print(img)
    imag = img_to_array(img)
    imaga = np.expand_dims(imag,axis=0)

    # print(stacked_image_array.shape)
    # stacked_image = Image.fromarray(stacked_image_array)
    stacked_image_array = cv2.resize(stacked_image_array, (256, 256))
    stacked_image_array = stacked_image_array.astype(np.float32)
    stacked_image_array = np.expand_dims(stacked_image_array,axis=0)
    
    # print(imaga.shape, stacked_image_array.shape)
    # print(imaga[0][0][0], stacked_image_array[0][0][0])

    last_conv_layer_name='conv2d_1' #enter the name of convolution layer
    # # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(stacked_image_array, heatmap_model, last_conv_layer_name)
    # print(stacked_image_array.shape)
    superimposed_img = heatmapImage(stacked_image_array[0], heatmap)

    return superimposed_img
    # plt.imsave("./Images/heatmap.png", superimposed_img)
    # print(superimposed_img.shape)
