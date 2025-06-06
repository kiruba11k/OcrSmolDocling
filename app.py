
import streamlit as st
import os
import time
import torch
import tempfile
import re
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import fitz  

HF_TOKEN = st.secrets["HF_TOKEN"]

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False

def check_dependencies():
    missing = []
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    if not docling_available:
        missing.append("docling-core")
    return missing

def process_single_image(image, prompt_text="Convert this page to docling."):
    if HF_TOKEN:
        login(token=HF_TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float32,
        ).to(device)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1024)  # Reduced for testing
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Clean the output
    doctags = doctags.replace("<end_of_utterance>", "").strip()
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    
    # Create a docling document
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    
    # Export as markdown
    md_content = doc.export_to_markdown()
    
    processing_time = time.time() - start_time
    
    return doctags, md_content, processing_time

def main():
    st.set_page_config(page_title="SmolDocling OCR App", layout="wide")
    st.title("SmolDocling OCR App - TSV Extraction")
    st.write("Upload multiple images to extract names, designations, and companies in tab-separated format.")

    if not HF_TOKEN:
        st.warning("HF_TOKEN not found in .env file. Authentication may fail.")

    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them to use this app.")
        st.info("Install with: pip install " + " ".join(missing_deps))
        st.stop()

    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        process_button = st.button("Process Images")

        if process_button:
            with st.spinner(f"Processing {len(images)} images..."):
                try:
                    results = []
                    for idx, image in enumerate(images):
                        st.write(f"Processing image {idx+1}/{len(images)}...")
                        doctags, md_content, proc_time, tsv_output = process_single_image(image)
                        results.append((image, doctags, md_content, proc_time, tsv_output))

                    for idx, (img, doctags, md_content, proc_time, tsv_output) in enumerate(results):
                        with st.expander(f"Image {idx+1} Results"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(img, caption=f"Image {idx+1}", width=250)
                                st.download_button(f"Download DocTags {idx+1}", doctags, file_name=f"output_{idx+1}.dt")
                                st.download_button(f"Download TSV {idx+1}", tsv_output, file_name=f"output_{idx+1}.tsv")
                                st.text_area(f"TSV Output {idx+1}", tsv_output, height=200)
                            with col2:
                                st.markdown(md_content)
                                st.download_button(f"Download Markdown {idx+1}", md_content, file_name=f"output_{idx+1}.md")

                    st.success("All images processed successfully.")
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()
