import streamlit as st
from transformers import ViTFeatureExtractor, ViTModel, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import torch

# Load pre-trained models and processors from your directory
vit_model_path = "./vit_model"
vit_processor_path = "./vit_processor"
gpt2_model_path = "./fine_tuned_gpt2_model"
gpt2_tokenizer_path = "./gpt2_tokenizer"

vit_processor = ViTFeatureExtractor.from_pretrained(vit_processor_path)
vit_model = ViTModel.from_pretrained(vit_model_path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_tokenizer_path)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

# Function to extract features using ViT
def extract_features(image, model, processor):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output

# Function to generate a caption using GPT-2 with a custom starting word
def generate_caption(features, gpt2_model, gpt2_tokenizer, max_length=50, start_word="Caption:"):
    if features is None:
        return "No features extracted for this image."
    
    prompt = f"{start_word} "  # Start the caption with the user-provided word or phrase
    inputs = gpt2_tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
    output = gpt2_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True
    )
    caption = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.title("Medical Image Captioning with ViT-SCL-ITP + GPT-2")

# Image Upload
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image...", type=["jpg", "png", "jpeg"])

# Input for custom starting word
start_word = st.text_input("Enter a starting word or phrase for the caption:", value="Findings:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a "Generate Caption" button
    if st.button("Generate Caption"):
        # Extract features and generate caption
        with st.spinner("Generating caption..."):
            features = extract_features(image, vit_model, vit_processor)
            caption = generate_caption(features, gpt2_model, gpt2_tokenizer, start_word=start_word)
        
        # Display the generated caption
        st.write("Generated Caption:")
        st.write(caption)
