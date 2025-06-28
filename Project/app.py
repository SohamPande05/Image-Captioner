import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import numpy as np
import gradio as gr

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image: np.ndarray):
    
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    text = "The image is about "
    # Process the image
    inputs = processor(images = raw_image, text = text, return_tensors = "pt")
    # Generate a caption for the image

    outputs = model.generate(**inputs, max_length=100 )

    # Decode the generated tokens to text and store it into `caption`

    caption = processor.decode(outputs[0], skip_special_tokens=True)

    
    return caption

iface = gr.Interface(

        fn = caption_image,

        # gr.Image() will return an array 
        inputs= gr.Image(type="numpy"),
        outputs= "text",
        title="Image Captioning",
        description= "This is a simple web app for generation for images using a trained model"
        
        )
iface.launch()


