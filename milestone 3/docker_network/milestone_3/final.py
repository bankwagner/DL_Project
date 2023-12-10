import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve
import gradio as gr
import gdown
import sys
import os
import time
import skimage.transform as skt
from PIL import Image
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
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    image = max_int_resize_and_normalize(image)
    
    if image.shape != (256, 256, 3):
        sys.exit(1)

    # clip pixel values as a safety measure
    image = np.clip(image, 0, 1)
         
    return image



#-----------------------Model loading-----------------------

def runUnet(img):#, progress=gr.Progress()):
    img = preprocess(img)
    unet = tf.keras.models.load_model("unet.h5", compile=False)
    img = unet.predict(np.expand_dims(img,0))
    img_pred = np.argmax(img, axis=3)[0,:,:]
    img_pred = np.stack((img_pred,)*3, axis=-1)
    class_segments = [np.where(img_pred == i + 1, i + 1, 0) for i in range(3)]
    pred_imgs=[]
    for i, segment in enumerate(class_segments):
        pred_imgs.append(Image.fromarray((segment*255).astype(np.uint8),'RGB'))

    img_pred_max = np.argmax(img, axis=3)[0,:,:]
    cls_sgm=[np.where(img_pred_max == i + 1, i + 1, 0) for i in range(3)]
    asd1=Image.fromarray((cls_sgm[0]*255).astype(np.uint8),'L')
    asd2=Image.fromarray((cls_sgm[1]*255).astype(np.uint8),'L')
    asd3=Image.fromarray((cls_sgm[2]*255).astype(np.uint8),'L')
    comb = Image.merge("RGB", (asd1, asd2, asd3))
    return comb, pred_imgs[0], pred_imgs[1], pred_imgs[2]

def runFPN(img):#, progress=gr.Progress()):
    img = preprocess(img)
    fpn = tf.keras.models.load_model("fpn.h5", compile=False)
    img = fpn.predict(np.expand_dims(img,0))
    img_pred = np.argmax(img, axis=3)[0,:,:]
    img_pred = np.stack((img_pred,)*3, axis=-1)
    class_segments = [np.where(img_pred == i + 1, i + 1, 0) for i in range(3)]
    pred_imgs=[]
    for i, segment in enumerate(class_segments):
        pred_imgs.append(Image.fromarray((segment*255).astype(np.uint8),'RGB'))
        
    img_pred_max = np.argmax(img, axis=3)[0,:,:]
    cls_sgm=[np.where(img_pred_max == i + 1, i + 1, 0) for i in range(3)]
    asd1=Image.fromarray((cls_sgm[0]*255).astype(np.uint8),'L')
    asd2=Image.fromarray((cls_sgm[1]*255).astype(np.uint8),'L')
    asd3=Image.fromarray((cls_sgm[2]*255).astype(np.uint8),'L')
    comb = Image.merge("RGB", (asd1, asd2, asd3))
    return comb, pred_imgs[0], pred_imgs[1], pred_imgs[2]


def runLinknet(img):#, progress=gr.Progress()):
    img = preprocess(img)
    linknet = tf.keras.models.load_model("linknet.h5", compile=False)
    img = linknet.predict(np.expand_dims(img,0))
    img_pred = np.argmax(img, axis=3)[0,:,:]
    img_pred = np.stack((img_pred,)*3, axis=-1)
    class_segments = [np.where(img_pred == i + 1, i + 1, 0) for i in range(3)]
    pred_imgs=[]
    for i, segment in enumerate(class_segments):
        pred_imgs.append(Image.fromarray((segment*255).astype(np.uint8),'RGB'))

    img_pred_max = np.argmax(img, axis=3)[0,:,:]
    cls_sgm=[np.where(img_pred_max == i + 1, i + 1, 0) for i in range(3)]
    asd1=Image.fromarray((cls_sgm[0]*255).astype(np.uint8),'L')
    asd2=Image.fromarray((cls_sgm[1]*255).astype(np.uint8),'L')
    asd3=Image.fromarray((cls_sgm[2]*255).astype(np.uint8),'L')
    comb = Image.merge("RGB", (asd1, asd2, asd3))
    return comb, pred_imgs[0], pred_imgs[1], pred_imgs[2]

def runEnsamble(img):#, progress=gr.Progress()):
    opt_weights = [0.97, 0.28, 0.64]
    loaded_models = []
    model1 = tf.keras.models.load_model("unet.h5", compile=False)
    model2 = tf.keras.models.load_model("linknet.h5", compile=False)
    model3 = tf.keras.models.load_model("fpn.h5", compile=False)
    loaded_models.append(model1)
    loaded_models.append(model2)
    loaded_models.append(model3)
    img = preprocess(img)
    test_preds=[]
    for model in loaded_models:
        test_preds.append(model.predict(np.expand_dims(img,0)))
    test_preds = np.array(test_preds)
    weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
    weighted_test_preds_ens = np.argmax(weighted_test_preds, axis=3)[0,:,:]
    img_pred = np.stack((weighted_test_preds_ens,)*3, axis=-1)
    class_segments = [np.where(img_pred == i + 1, i + 1, 0) for i in range(3)]
    pred_imgs=[]
    for i, segment in enumerate(class_segments):
        pred_imgs.append(Image.fromarray((segment*255).astype(np.uint8),'RGB'))
        
    img_pred_max = np.argmax(weighted_test_preds, axis=3)[0,:,:]
    cls_sgm=[np.where(img_pred_max == i + 1, i + 1, 0) for i in range(3)]
    asd1=Image.fromarray((cls_sgm[0]*255).astype(np.uint8),'L')
    asd2=Image.fromarray((cls_sgm[1]*255).astype(np.uint8),'L')
    asd3=Image.fromarray((cls_sgm[2]*255).astype(np.uint8),'L')
    comb = Image.merge("RGB", (asd1, asd2, asd3))
    return comb, pred_imgs[0], pred_imgs[1], pred_imgs[2]

with gr.Blocks() as demo:
  gr.Markdown("""
              # Model Ensable project for the Deep Learning course at BME by the team AIvengers
              """)
  name = gr.Image(label="Upload an image", type="numpy")
  with gr.Tab("Unet"):
    with gr.Row():
      model_1 = gr.Image(height=256, width=256)
      model_1_1 = gr.Image(height=256, width=256, label="Left Ventricle", show_label=True)
      model_1_2 = gr.Image(height=256, width=256, label="Right Ventricle", show_label=True)
      model_1_3 = gr.Image(height=256, width=256, label="Myocardium", show_label=True)
    model_1_btn = gr.Button("Model Unet Futtatása")
    gr.Markdown("""
    unet - ~290 MB
                
          - training: batch_size 8, learning_rate 0.0001, epochs 30 --> takes about 15 mins
                
          - evaluation: Mean IoU = 0.8295, Mean F1 Score = 0.9459
                
          - chatgpt: UNet is a convolutional neural network (CNN) architecture commonly used for image segmentation tasks. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. The name "UNet" comes from its U-shaped architecture.
          
          Here are the key features of the UNet architecture:

            Encoder-Decoder Structure: UNet consists of two main parts: an encoder and a decoder. The encoder is responsible for capturing the contextual information from the input image, while the decoder is responsible for spatial localization.

            Contracting Path (Encoder): The encoder is designed as a series of convolutional and pooling layers that progressively reduce the spatial resolution of the input image while increasing the depth of features. This is sometimes referred to as the "contracting path" because it compresses the input information into a more abstract representation.

            Expansive Path (Decoder): The decoder, or expansive path, is designed to upsample the feature map to the original input resolution. It uses transposed convolutions (also known as deconvolutions or upsampling) to gradually recover the spatial information.

            Skip Connections: One notable feature of UNet is the use of skip connections. Skip connections connect corresponding layers between the encoder and decoder. These connections allow the network to retain high-resolution information during the upsampling process, helping to improve segmentation performance.

            Final Convolutional Layer: The architecture typically ends with a final convolutional layer that produces the segmentation mask. This layer usually has a number of output channels equal to the number of classes in the segmentation task.

            UNet is widely used in medical image segmentation, such as segmenting organs or lesions in medical scans. Its architecture has been found to be effective in capturing both local and global features, and the skip connections help in preserving fine details during the upsampling process. Various modifications and improvements to the original UNet architecture have been proposed over time to address specific challenges in different segmentation tasks.

""")
  with gr.Tab("FPN"):
    with gr.Row():
      model_2 = gr.Image(height=256, width=256)
      model_2_1 = gr.Image(height=256, width=256, label="Left Ventricle", show_label=True)
      model_2_2 = gr.Image(height=256, width=256, label="Right Ventricle", show_label=True)
      model_2_3 = gr.Image(height=256, width=256, label="Myocardium", show_label=True)
    model_2_btn = gr.Button("Model FPN Futtatása")
    gr.Markdown("""
    linknet - ~250 MB
                
              - training: batch_size 8, learning_rate 0.0001, epochs 30 --> takes about 15 mins
                
              - evaluation: Mean IoU = 0.8170, Mean F1 Score = 0.9405
                
              - chatgpt: LinkNet is another convolutional neural network (CNN) architecture designed for image segmentation tasks. It was proposed by Alexander G. Buslaev, Alexey I. Zhidkov, Valentin V. Iglovikov, and Alexey A. Shvets in 2017. Similar to UNet, LinkNet is used for semantic segmentation, where the goal is to assign a label to each pixel in an input image.
             
              Here are some key features of the LinkNet architecture:

                Encoder-Decoder Architecture: LinkNet, like UNet, follows an encoder-decoder structure. The encoder is responsible for extracting features from the input image, and the decoder is responsible for generating the segmentation mask.

                Skip Connections with Link Modules: One of the distinctive features of LinkNet is the use of link modules, which are skip connections with residual connections. These links connect corresponding layers between the encoder and decoder, similar to skip connections in UNet. However, in LinkNet, residual connections are incorporated into the links to facilitate smoother information flow.

                Residual Blocks in Encoder: LinkNet utilizes residual blocks in the encoder, which are inspired by the residual networks (ResNet) architecture. Residual blocks help address the vanishing gradient problem, allowing the network to train more effectively, especially when dealing with deep architectures.

                Spatial Attention Module: LinkNet includes a spatial attention module, which helps the network focus on more relevant parts of the input image during the segmentation process. This attention mechanism aids in capturing long-range dependencies in the image.

                Batch Normalization: Batch normalization is used throughout the network to normalize activations and accelerate training. This normalization technique helps with the stability and speed of convergence during the training process.

                LinkNet has been applied to various segmentation tasks, including medical image segmentation and object detection. Its modular design, use of skip connections with residual connections, and attention mechanisms contribute to its effectiveness in capturing both local and global contextual information in images. The combination of these features helps improve segmentation accuracy, particularly in cases where capturing fine details and preserving spatial relationships are crucial.

""")
  with gr.Tab("LinkNet"):
    with gr.Row():
      model_3 = gr.Image(height=256, width=256)
      model_3_1 = gr.Image(height=256, width=256, label="Left Ventricle", show_label=True)
      model_3_2 = gr.Image(height=256, width=256, label="Right Ventricle", show_label=True)
      model_3_3 = gr.Image(height=256, width=256, label="Myocardium", show_label=True)
    model_3_btn = gr.Button("Model Link Net Futtatása")
    gr.Markdown("""
    fpn - ~280 MB
                
            - training: batch_size 8, learning_rate 0.0001, epochs 20 --> takes about 20 mins
                
            - evaluation: Mean IoU = 0.8225, Mean F1 Score = 0.9396
                
            - chatgpt: Feature Pyramid Network (FPN) is another popular convolutional neural network (CNN) architecture, particularly used for object detection tasks. FPN was introduced by Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan in a paper titled "Feature Pyramid Networks for Object Detection," published in 2017.
           
            Here are the key features of the Feature Pyramid Network (FPN) architecture:

              Multi-scale Feature Pyramid: FPN addresses the challenge of object detection at multiple scales by creating a feature pyramid. The pyramid consists of feature maps at different spatial resolutions, allowing the network to capture information at various scales. This is crucial for detecting objects of different sizes in an image.

              Bottom-up and Top-down Architecture: FPN combines a bottom-up pathway (similar to a standard convolutional network) with a top-down pathway. The bottom-up pathway involves the typical convolutional layers that extract high-level features from the input image. The top-down pathway involves upsampling and merging high-resolution features from the lower levels of the pyramid to the higher levels, enhancing the network's ability to detect objects at different scales.

              Feature Fusion: FPN incorporates feature fusion to combine information from different levels of the pyramid. This fusion is achieved by adding feature maps from the top-down pathway to the corresponding feature maps from the bottom-up pathway. The fusion helps improve the localization of objects across various scales.

              Pyramid Pooling: FPN uses a technique called pyramid pooling to pool features at multiple scales. This involves pooling features from different pyramid levels and concatenating them, providing a more comprehensive representation of the image.

              Object Detection Heads: FPN is often used in conjunction with object detection frameworks, such as Faster R-CNN. The multi-scale feature pyramid is connected to the Region Proposal Network (RPN) and the object detection heads, facilitating the generation of region proposals and the final object detections.

              FPN has proven effective in improving the performance of object detection models, especially in scenarios where objects have varying sizes. It has become a standard component in many state-of-the-art object detection architectures.

""")
  with gr.Tab("Ensamble"):
    with gr.Row():
      model_e = gr.Image(height=256, width=256)
      model_e_1 = gr.Image(height=256, width=256, label="Left Ventricle", show_label=True)
      model_e_2 = gr.Image(height=256, width=256, label="Right Ventricle", show_label=True)
      model_e_3 = gr.Image(height=256, width=256, label="Myocardium", show_label=True)
    model_e_btn = gr.Button("Model Ensamble Futtatása")
    
  with gr.Accordion("Project description"):
    gr.Markdown("In this project, you'll dive into the idea of using multiple models together, known as model ensembles, to make our deep learning solutions more accurate. They are a reliable approach to improve accuracy of a deep learning solution for the added cost of running multiple networks. Using ensembles is a trick that's widely used by the winners of AI competitions. Task of the students: explore approaches to model ensemble construction for semantic segmentation, select a dataset (preferentially cardiac MRI segmentation, but others also allowed), find an open-source segmentation solution as a baseline for the selected dataset and test it. Train multiple models and construct an ensemble from them. Analyse the improvements, benefits and added costs of using an ensemble.")


  #upload_btn.click(fn=bye, inputs=name, outputs=[model_1, model_3])
  model_1_btn.click(fn=runUnet, inputs=name, outputs=[model_1, model_1_1, model_1_2, model_1_3])
  model_2_btn.click(fn=runFPN, inputs=name, outputs=[model_2, model_2_1, model_2_2, model_2_3])
  model_3_btn.click(fn=runLinknet, inputs=name, outputs=[model_3, model_3_1, model_3_2, model_3_3])
  model_e_btn.click(fn=runEnsamble, inputs=name, outputs=[model_e, model_e_1, model_e_2, model_e_3])

  
    
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", debug=True)  
