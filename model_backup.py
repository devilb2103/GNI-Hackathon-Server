# from base64 import encodebytes
# import io
# import os
# from flask import Response, jsonify, request
# os.environ["SM_FRAMEWORK"] = "tf.keras"
# from PIL import Image, ImageOps
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array, array_to_img
# import tensorflow as tf
# import segmentation_models
# import matplotlib.pyplot as plt
# import cv2
# from heatmap import heatMapArr
# import inspect


# classification_model = load_model("./Models/TCGA_Multitask_res50_4pred_1.h5")
# segmentation_model = tf.keras.models.load_model("./Models/brain tumour segmentation model.h5",compile=False)

# def imageArr_to_bytes(imgArr: np.ndarray) -> str:
#     try:
#         image = Image.fromarray(imgArr)
#         img_bytes = io.BytesIO()
#         image.save(img_bytes, format='PNG')
#         encoded_img = encodebytes(img_bytes.getvalue()).decode('ascii')
#         return encoded_img
#     except Exception as e:
#         call_name = inspect.stack()[0][3]
#         error_message = f"(function: {call_name}) Array to bytes converion Error -> {e}"
#         raise Exception(error_message)

# def classifyImage(stacked_image_array: np.ndarray) -> list:
#     try:
#         stacked_image = Image.fromarray(stacked_image_array)
#         stacked_image_Expanded = np.expand_dims(stacked_image, axis=0)
#         model = classification_model.predict(stacked_image_Expanded)
#         class_arr = [['g3','g4'],
#                     ['mutant','wild'],
#                     ['codeleted', 'non - codeleted'],
#                     ['methylated', 'unmethylated']]
#         predictions = {"Grade":"","IDH Type":"","1p/19q":"", "MGMT":""}
#         for i in range(4):
#             predictions[list(predictions.keys())[i]] = class_arr[i][round(model[i][0][0])]
#         return predictions
#     except Exception as e:
#         call_name = inspect.stack()[0][3]
#         error_message = f"(function: {call_name}) Image Classification Error -> {e}"
#         raise Exception(error_message)


# def classify_segment_heatmap(files: list) -> int:
#     try:
#         # load images as bytes in memory for stacking
#         images = []
#         for file in files:
#             img = Image.open(file)
#             img_bytes = io.BytesIO()
#             # save it to bytes stream
#             img.save(img_bytes, format='PNG')
#             # append bytes stream to images array
#             images.append(load_img(img_bytes, target_size=(128,128,1), color_mode="grayscale"))
#         # stack images that will go into 
#         # classification and segmentation model
#         stacked_image_array = np.stack(images, axis=-1)

#         # make classifications
#         print("Performing Image Classification ------------------------------------------------------------------------------")
        
#         classifications = classifyImage(stacked_image_array)

#         # get segmentation bytes
#         # save flair image temporarily for segmentation
#         plt.imsave("flair.png", np.asarray(images[0]))
#         print("Performing Image Segmentation ------------------------------------------------------------------------------")
#         segmented_bytes = segmentImage()

#         # get heatmap bytes
#         print("Generating Tumor Heatmap ------------------------------------------------------------------------------")
#         heatmap_bytes = generateHeatmap(stacked_image_array)

#         # response data
#         res = jsonify({"status": 200, "message": {"classifications": classifications, "imageBytes": [segmented_bytes, heatmap_bytes]}})
#         return res
#     except Exception as e:
#         res = {"status": 400, "message": f"classify_segment_heatmap Error: {e}"}
#         print(res)
#         return res

# def segmentImage() -> str:
#     try:
#         image = cv2.imread('./flair.png', 1)
#         image = cv2.resize(image,(256,256))
#         image_normalized = image/255
#         image_expanded = np.expand_dims(image_normalized,axis=0)
#         seg_image = image_expanded.astype(np.float32)
#         # print(seg_image.shape)

#         predicted = segmentation_model.predict(seg_image)
#         predicted_filtered = np.argmax(predicted, axis=-1)
#         # plt.imsave("./Images/predicted.png", predicted_filtered[0])

        
#         predicted_filtered = (predicted_filtered[0] * 255).astype(np.uint8)
#         segmented_bytes = imageArr_to_bytes(predicted_filtered)
#         return segmented_bytes
#     except Exception as e:
#         call_name = inspect.stack()[0][3]
#         error_message = f"(function: {call_name}) Segment Image Error -> {e}"
#         raise Exception(error_message)

# def generateHeatmap(stacked_image_array: np.ndarray) -> str:
#     try:
#         heatmap = heatMapArr(stacked_image_array)
#         heatmap_bytes = imageArr_to_bytes(heatmap)
#         return heatmap_bytes
#     except Exception as e:
#         call_name = inspect.stack()[0][3]
#         error_message = f"(function: {call_name}) Heatmap generation Error -> {e}"
#         raise Exception(error_message)