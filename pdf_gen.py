from flask import send_file
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import datetime
from PIL import Image
import numpy as np
import os
import inspect

# def clampImageDimensions(image):
#     return [size_x, size_y]

def replace_text_in_pdf(input_pdf_path, output_pdf_path, replacements, offsets):
    try:
        pdf_document = fitz.open(input_pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            for search_text, replace_text in replacements.items():
                text_instances = page.search_for(search_text)
                for inst in reversed(text_instances):
                    x1, y1, x2, y2 = inst  # Get the bounding box of the old text
                    _x1, _y1, _x2, _y2 = x1 + offsets[0], y1 + offsets[1], x2 + offsets[2], y2 + offsets[3]
                    if(type(replace_text) == list):
                        # rect = fitz.Rect(x1, y1, x2, y2)
                        # page.draw_rect((x1, y1, x2, y2), color=(1, 1, 1), fill=(1, 1, 1), overlay=True)  # Draw a filled rectangle to cover the old text

                        image_rect = fitz.Rect(_x1 + offsets[0] - replace_text[2][0], _y1 + offsets[1] - replace_text[2][1], _x1 + replace_text[2][2] + replace_text[1][0], _y1 +replace_text[2][3] + replace_text[1][1])
                        page.insert_image(image_rect, filename=replace_text[0])
                        # image stuff
                    else:
                        rect = fitz.Rect(x1, y1, x2, y2)
                        page.draw_rect((x1, y1, x2, y2), color=(1, 1, 1), fill=(1, 1, 1), overlay=True)  # Draw a filled rectangle to cover the old text
                        page.insert_text((_x1, _y1), replace_text, fontsize=13, color=(0, 0, 0))  # Insert the new text

        pdf_document.save(output_pdf_path)
        pdf_document.close()
    except Exception as e:
        call_name = inspect.stack()[0][3]
        error_message = f"(function: {call_name}) PDF Writing Error -> {e}"
        raise Exception(error_message)

def generatePDF_internal(segmented_image_path: str, heatmap_image_path: str, 
                         reportData: dict) -> None:
    # Example usage:
    try:
        input_pdf_path = './BRAIN TUMOR ANALYSIS REPORT.pdf'
        output_pdf_path = './BRAIN TUMOR ANALYSIS REPORT - Processed.pdf'

        replacements = {
            
            '<analysis_date>': f'{datetime.datetime.now().day}/{datetime.datetime.now().month}/{datetime.datetime.now().year}'.replace("-"," / "),
            '<image_type>': 'MRI - T1, T2 and FLAIR',
            '<roi>': [segmented_image_path, [230, 230], [4.1,22,4.7,10]],
            '<gradcam>': [heatmap_image_path, [230, 229.5], [4.1,22.7,4.7,10]],

            # Add more key-value pairs for other text replacements
            # use "/" for image path, automatically understands that its an image replacement
        }

        for i in reportData.keys():
            replacements[i] = reportData[i]

        offsets = [0, 10.5, 0, 13]
        replace_text_in_pdf(input_pdf_path, output_pdf_path, replacements, offsets)
    except Exception as e:
        call_name = inspect.stack()[0][3]
        error_message = f"(function: {call_name}) PDF Writing Initialization Error -> {e}"
        raise Exception(error_message)

def generatePDF(patientData: dict, images):
    try:
        paths = []
        if(not(os.path.exists("temp"))):
            os.mkdir("temp")
        for i in images:
            img = Image.open(i)
            path = f"temp/{i.filename}.png"
            paths.append(path)
            plt.imsave(path, np.asarray(img))
        
        classificationData = eval((patientData['classifications']).replace('{', '{"').replace('}', '"}').replace(', ', '", "').replace(': ', '": "'))
        print(paths)
        print(classificationData)

        # add report data
        data = {
            '<report_number>': patientData["<report_number>"],
            '<patient_id>': patientData["<patient_id>"],
            '<patient_age>': patientData["<patient_age>"],
            '<scan_date>': patientData["<scan_date>"],
            '<pathology>': "",
            '<grade>': classificationData['Grade'],
            "<idh>": classificationData['IDH_Type'],
            "<mgmt>": classificationData['MGMT'],
            "<1p/19q>": classificationData['1p/19q'],
        }
        if(data['<grade>'] == "G3"):
            data['<pathology>'] = "Astrocytoma"
        else:
            data['<pathology>'] = "Glioblastoma"

        generatePDF_internal(paths[0], paths[1], data)
    except Exception as e:
        call_name = inspect.stack()[0][3]
        error_message = f"(function: {call_name}) PDF Data Initializater Error -> {e}"
        raise Exception(error_message)

def getGeneratedPDF(classifications, imageFiles):
    try:
        if(os.path.exists("./BRAIN TUMOR ANALYSIS REPORT - Processed.pdf")):
            os.remove("./BRAIN TUMOR ANALYSIS REPORT - Processed.pdf")
        res = None
        
        generatePDF(classifications, imageFiles)

        max_wait_iterator = 0
        while True:
            max_wait_iterator += 1
            if(max_wait_iterator > 5000):
                break
            if(os.path.exists("./BRAIN TUMOR ANALYSIS REPORT - Processed.pdf")):
                res = send_file('./BRAIN TUMOR ANALYSIS REPORT - Processed.pdf', as_attachment=True, 
                download_name="BRAIN_TUMOR_ANALYSIS_REPORT.pdf", mimetype="application/pdf")
                break
        return res
    except Exception as e:
        call_name = inspect.stack()[0][3]
        error_message = f"(function: {call_name}) PDF API Call Reciever Error -> {e}"
        raise Exception(error_message)