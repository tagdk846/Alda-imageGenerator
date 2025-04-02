import torch
from diffusers import FluxPipeline
import streamlit as st
from PIL import Image
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# Load the model
@st.cache_resource
def load_model():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

pipe = load_model()

# Streamlit UI
st.title("AI Image Generator")
st.write("Generate images using the FLUX.1-dev model.")

# User input
prompt = st.text_input("Enter your prompt:", "A cat holding a sign that says Hello World")

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=50
        ).images[0]

        # Display the image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Save the image
        image.save("generated_image.png")
        st.success("Image saved as generated_image.png")
        
