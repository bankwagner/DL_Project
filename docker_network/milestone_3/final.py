import tensorflow as tf
import numpy as np
from urllib.request import urlretrieve
import gradio as gr

urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5", "mnist-model.h5")
model = tf.keras.models.load_model("mnist-model.h5")

# def recognize_digit(image):
#     image = image.reshape(1, -1)  
#     prediction = model.predict(image).tolist()[0]
#     return {str(i): prediction[i] for i in range(10)}

# output_component = gr.Label(num_top_classes=5)

# gr.Interface(fn=recognize_digit, 
#              inputs="image", 
#              outputs=output_component,
#              title="Model Ensamble",
#              description="Upload an image").launch(server_name="0.0.0.0")


def bye(name):
    return "Bye " + name + "!"

def greet(name):
    return "Hello " + name + "!"

def preprocess(img, progress=gr.Progress()):
    progress(0, desc="Preprocess starting")
    img = img.reshape(1, -1)  
    prediction = model.predict(img).tolist()[0]
    progress.update(0.5)
    return {str(i): prediction[i] for i in range(10)}

#demo = gr.Interface(fn=greet, inputs="text", outputs=output_component)

with gr.Blocks() as demo:
  gr.Markdown("Model Ensable project for the Deep Learning course at BME by the team AIvengers")
  name = gr.Textbox(label="Upload an image")
  #upload_btn = gr.Button("Get Segmentation")
  with gr.Tab("Model 1"):
    model_1 = gr.Label()
    #greeting=gr.Image()    
    model_1_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("Model 2"):
    model_2 = gr.Label()
    model_2_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("Model 3"):
    model_3 = gr.Label()
    model_3_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("Model 4"):
    model_4 = gr.Label()
    model_4_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("Model 5"):
    model_5 = gr.Label()
    model_5_btn = gr.Button("Model Unet Futtatása")
  with gr.Tab("Ensamble"):
    model_e = gr.Label()
    model_e_btn = gr.Button("Model Unet Futtatása")
    
  with gr.Accordion("Project description"):
    gr.Markdown("Description")


  #upload_btn.click(fn=bye, inputs=name, outputs=[model_1, model_3])
  model_1_btn.click(fn=bye, inputs=name, outputs=[model_1])
  model_2_btn.click(fn=bye, inputs=name, outputs=[model_2])
  model_3_btn.click(fn=bye, inputs=name, outputs=[model_3])
  model_4_btn.click(fn=bye, inputs=name, outputs=[model_4])
  model_5_btn.click(fn=bye, inputs=name, outputs=[model_5])
  model_e_btn.click(fn=bye, inputs=name, outputs=[model_e])

  
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")  




  