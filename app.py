import streamlit as st
import os
import time
import torch
from PIL import Image
from dotenv import load_dotenv
import re

# PDF image extraction support
import fitz  # PyMuPDF

# Load .env variables
load_dotenv()
HF_TOKEN = st.secrets["HF_TOKEN"]

# Dependency flags
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


# Dependency check
def check_dependencies():
    missing = []
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    if not docling_available:
        missing.append("docling-core")
    return missing


# TSV extractor from DoclingDocument
def extract_tsv_from_doc(doc: DoclingDocument) -> str:
    """
    Extract Name, Designation, Company fields from DoclingDocument into TSV format.
    """
    tsv_rows = []
    header = ["Name", "Designation", "Company"]
    tsv_rows.append("\t".join(header))

    for element in doc.elements:
        if hasattr(element, "fields"):
            name = element.fields.get("name", "").strip()
            designation = element.fields.get("designation", "").strip()
            company = element.fields.get("company", "").strip()

            if name or designation or company:
                row = [name, designation, company]
                tsv_rows.append("\t".join(row))

    return "\n".join(tsv_rows)


# Image processing function
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

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt",truncation=True).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
    doctags = doctags.replace("<end_of_utterance>", "").strip()

    # Construct Docling object
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    md_content = doc.export_to_markdown()

    # Extract structured TSV
    tsv_output = extract_tsv_from_doc(doc)

    processing_time = time.time() - start_time
    return doctags, md_content, processing_time, tsv_output


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
