import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from model import classify_segment_heatmap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pdf_gen import getGeneratedPDF

app = Flask(__name__)   
CORS(app)


@app.route('/health')
def health():
    return {"status": 200, "message": "server is up and running"}


@app.route('/predict', methods=['POST'])
def predict():
    imageFiles = request.files.getlist('imagefiles')

    predictions = classify_segment_heatmap(imageFiles)
    return predictions

@app.route('/getPDF', methods=['POST'])
def getPDF():
    # extra = {
    #     "msg":"lol"
    # }
    # res.headers['data'] = jsonify(extra)
    classifications = request.form.to_dict()
    imageFiles = request.files.getlist('imagefiles')
    res = getGeneratedPDF(classifications, imageFiles)
    # ------------------------------------------------------------------------------
    # write container functtion for this and retorn error if there is error
    return res
