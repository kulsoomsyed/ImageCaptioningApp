# Image Captioning App

Small Streamlit demo app for Chest X-ray image caption generation. This app was created to demonstrate one of the best-performing model variants from the MSc project: **ViT-SCL-ITP + GPT-2**
Allows users to upload a chest X-ray image and generate a caption using the trained model.

---

## Purpose

- Not a full production application.
- Simple interface for testing caption generation.
- Built using Streamlit.
- Intended for quick testing/demo of the model output.

---

## Model Used

- **Model**: ViT-SCL-ITP + GPT-2
- **Image encoder**: Vision Transformer
- **Learning approach**: Image-Text Pair Supervised Contrastive Learning
- **Caption generator**: GPT-2

---

## Installation

Make sure Streamlit is installed in your Python environment.

```bash
pip install streamlit
```

## Running the App

Open CMD/terminal and navigate to the folder where app.py is located.

Then run:
```
streamlit run app.py
```
The app will open in a new browser tab.

Upload and Test Images
- Upload a chest X-ray image in MIMIC-CXR JPG format.
- Click Generate Caption.
- The app will generate a caption for the uploaded image.

## Dataset
- Dataset used: MIMIC-CXR / MIMIC-CXR-JPG
- Dataset available through PhysioNet.
- Access requires a PhysioNet account and completion of the required training/course.
- Full dataset is not included in this repository.

---

## NOTE
This repository only contains the Streamlit demo app. The full experimental codes, model variants, generated CSV files, and evaluation notebooks are available here:
https://github.com/kulsoomsyed/SCLDIPCXR

That repository includes:
Pre-executed codes
Generated caption CSV files
Evaluation files
Different model variant folders

In the pre-executed experiments, the original split used:
4057 training images
50 test images

---

## Repository Structure

```text
ImageCaptioningApp/
│
├── app.py
├── README.md
│
├── fine_tuned_gpt2_model/
│   ├── config.json
│   └── generation_config.json
│
├── gpt2_tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
│
└── vit_model/
    └── config.json
