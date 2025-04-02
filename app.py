import torch
from diffusers import FluxPipeline
import streamlit as st
from PIL import Image

# Write the Streamlit app code to a file
streamlit_code = """
import torch
from diffusers import FluxPipeline
import streamlit as st
from PIL import Image

# Load the model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Streamlit UI Setup
st.title("AI Image Generator")
st.write("Generate images from text prompts using a pre-trained model.")

# Text input for the prompt
prompt = st.text_input("Enter your prompt:", "A cat holding a sign that says hello world")

# Slider for controlling guidance scale
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=3.5)

# Number of inference steps slider
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=100, value=50)

# Generate the image when the button is clicked
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image using the model
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        
        # Display the image in the UI
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Save the image
        image.save("generated_image.png")
        st.success("Image saved as generated_image.png")
"""

# Save the Streamlit code to a file
with open("streamlit_app.py", "w") as f:
    f.write(streamlit_code)
  
