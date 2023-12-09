import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve
import gradio as gr
import gdown
import sys
import os
import skimage.transform as skt
from PIL import Image

#-----------------------Model downloading-----------------------
gdown.download("https://drive.google.com/file/d/1J8hIRo-GT6XrkWemIum8DjRisjff9lca", "unet-pre_trained.h5", verify = False)
gdown.download("https://drive.google.com/file/d/1_C-tAH4wee_fkPdO8INrvZ2IGpDW8xh0", "fpn-pre_trained.h5", verify = False)
gdown.download("https://drive.google.com/file/d/1H0NnKsYLr8l5o1Xe5vfUiZj4Y6STWUKB", "linknet-pre_trained.h5", verify = False)
#urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5", "ensamble.h5")



#-----------------------Preprocessing-----------------------

def max_intenzity(image):
    """
    Normalize the intensity values of an image to the range [0, 255].

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized, max intenzity image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    min_max_norm_img = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return min_max_norm_img


def normalize(image):
    """
    Normalize the intensity values of an image to the range [0, 1].

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = ((image - min_val) / (max_val - min_val))
    return norm_img


def max_int_resize_and_normalize(numpy_img):
    """
    Resize, normalize, and convert the input image 

    Args:
        numpy_img : The input image.
        

    Returns:
        resized and normalized image
    """
    
    resized_norm_data = skt.resize(max_intenzity(numpy_img), (256,256,3), order=1, preserve_range=False,anti_aliasing=True)
    resized_norm_data = normalize(resized_norm_data)

    return resized_norm_data

def preprocess(image):
    print(" - - - preprocessing started - - - ")

    # Set environment variable for the SM_FRAMEWORK
    os.environ["SM_FRAMEWORK"] = "tf.keras"

    # if len(sys.argv) != 2:
    #     print("Usage: python prep_test.py image_path")
    # else:
    #     # Get the image file path from command-line argument
    #     image = sys.argv[1]

    # if the image are not in PIL.Image format, convert them 
    #if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
    #    image = tf.keras.preprocessing.image.load_img(image)
    image = max_int_resize_and_normalize(image)
    
    #if the image are in PIL format, convert them to numpy format
    #elif isinstance(image, Image.Image):
    #    image = max_int_resize_and_normalize(image)
    
    if image.shape != (256, 256, 3):
        print("Error: wrong image size.")
        sys.exit(1)

    # clip pixel values as a safety measure
    image = np.clip(image, 0, 1)
    
    # Print information about the preprocessed data
    if image is not None:
        print("Image converted to NumPy array successfully.")
        print("Data type of the array:", image.dtype)
        print("test_image:\t shape: ", image.shape, "\tmin: ", np.min(image), "max: ", np.max(image))
     
    # Save the preprocessed data to disk
    print(" - - - preprocessing finished - - - ")
    return image



#-----------------------Model loading-----------------------

def runUnet(img):#, progress=gr.Progress()):
    print(" - - - model preproc started - - - ")
    img = preprocess(img)
    print(" - - - model preproc finished - - - ")
    print(" - - - model loading started - - - ")
    unet = tf.keras.models.load_model("unet-pre_trained.h5")
    print(" - - - model loading finished - - - ")
    img = unet.predict(img)
    print(" - - - model prediction finished - - - ")
    return unet

def runFPN(img):#, progress=gr.Progress()):
    fpn = tf.keras.models.load_model("fpn-pre_trained.h5")
    img = fpn.predict(img)
    return img


def runLinknet(img):#, progress=gr.Progress()):
    linknet = tf.keras.models.load_model("linknet-pre_trained.h5")
    return img

def runEnsamble(img):#, progress=gr.Progress()):
    ensamble = tf.keras.models.load_model("ensamble.h5")
    return img

def bye(name):
    return "Bye " + name + "!"

def greet(name):
    return "Hello " + name + "!"

# def preprocess(img, progress=gr.Progress()):
#     progress(0, desc="Preprocess starting")
#     img = img.reshape(1, -1)  
#     prediction = model.predict(img).tolist()[0]
#     progress.update(0.5)
#     return {str(i): prediction[i] for i in range(10)}

#demo = gr.Interface(fn=greet, inputs="text", outputs=output_component)

with gr.Blocks() as demo:
  gr.Markdown("""
              # Model Ensable project for the Deep Learning course at BME by the team AIvengers
              """)
  name = gr.Image(label="Upload an image", type="pil")
  #upload_btn = gr.Button("Get Segmentation")
  with gr.Tab("Unet"):
    model_1 = gr.Image()
    model_1_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("FPN"):
    model_2 = gr.Image()
    model_2_btn = gr.Button("Model FPN Futtatása")
  with gr.Tab("LinkNet"):
    model_3 = gr.Image()
    model_3_btn = gr.Button("Model Link Net Futtatása")
  with gr.Tab("Ensamble"):
    model_e = gr.Image()
    model_e_btn = gr.Button("Model Ensamble Futtatása")
    
  with gr.Accordion("Project description"):
    gr.Markdown("In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep learning solutions more accurate. They are a reliable approach to improve accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions. Task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset and test it. Train multiple models and construct an ensemble from them. Analyse the improvements, benefits and added costs of using an ensemble.")


  #upload_btn.click(fn=bye, inputs=name, outputs=[model_1, model_3])
  model_1_btn.click(fn=runUnet, inputs=name, outputs=[model_1])
  model_2_btn.click(fn=runFPN, inputs=name, outputs=[model_2])
  model_3_btn.click(fn=runLinknet, inputs=name, outputs=[model_3])
  model_e_btn.click(fn=runEnsamble, inputs=name, outputs=[model_e])

  
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", debug=True)  




  