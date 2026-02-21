#!/usr/bin/env python3
"""
Automated PDF processor (FLOWCHART MODE ONLY)

- Takes start page and end page as input
- For each page:
  - Crops margins
  - Renders image
  - Sends to AWS Bedrock (Claude Vision)
  - Saves output in existing format
- No raw text mode
- No user interaction
"""

import base64
import json
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import boto3

# ------------------------ CONFIG ------------------------

PDF_PATH = "ovarian.pdf"
OUTPUT_PATH = "nccn_ovarian_output.txt"
AWS_REGION = "us-east-1"

MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"

# Default margins (PDF points)
MARGIN_LEFT = 0
MARGIN_RIGHT = 0
MARGIN_TOP = 60
MARGIN_BOTTOM = 30

# ------------------------------------------------------


def extract_clip_rect(pdf_path, page_num):
    """Return clipped rectangle after applying margins."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    rect = page.rect

    clipped_rect = fitz.Rect(
        rect.x0 + MARGIN_LEFT,
        rect.y0 + MARGIN_TOP,
        rect.x1 - MARGIN_RIGHT,
        rect.y1 - MARGIN_BOTTOM
    )

    doc.close()
    return clipped_rect


def render_clipped_image(pdf_path, page_num, clip_rect):
    """Render page image and crop to clipped rectangle."""
    page_img = convert_from_path(
        pdf_path,
        first_page=page_num + 1,
        last_page=page_num + 1
    )[0]

    pdf_doc = fitz.open(pdf_path)
    pdf_page = pdf_doc[page_num]
    dpi_scale = page_img.size[1] / pdf_page.rect.height
    pdf_doc.close()

    crop_box = (
        int(clip_rect.x0 * dpi_scale),
        int(clip_rect.y0 * dpi_scale),
        int(clip_rect.x1 * dpi_scale),
        int(clip_rect.y1 * dpi_scale)
    )

    return page_img.crop(crop_box)


def encode_image_to_base64(pil_img):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def invoke_claude_image(prompt_text, b64_image):
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=json.dumps(payload)
    )

    raw = response["body"].read().decode("utf-8")
    data = json.loads(raw)
    return data.get("content", [{}])[0].get("text", "")


def append_to_output(page_num, content):
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== PAGE {page_num} ===\n")
        f.write(content.strip() + "\n")
        f.write("=== END PAGE ===\n")


def process_pages(pdf_path, start_page, end_page):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    start_idx = max(1, start_page) - 1
    end_idx = min(end_page, total_pages) - 1

    prompt_text = (
        "You are given an image of a PDF page that may contain a flowchart or a table or diagram. "
        "Convert this page of guidelines that has flowchart into paragraph that can be used for RAG. "
        "Convert the visual guidance into a single clear, concise paragraph suitable for RAG ingestion. "
        "Do NOT add facts not present in the image or invent steps. "
        "If it's a paragraph, retain it as is and remove superscripts only. "
        "If it's a table, convert it into JSON format. "
        "If drugs are mentioned based on gene deletion or mutation (e.g., EGFR exon), include that. "
        "Include blue references in brackets shown on the right side of the flowchart. "
        "Do NOT include category notes or reference section labels. "
        "Include the page number (alphanumeric) shown at the bottom-right of the PDF as a new line."
    )

    for page in range(start_idx, end_idx + 1):
        print(f"Processing page {page + 1}...")

        try:
            clip_rect = extract_clip_rect(pdf_path, page)
            image = render_clipped_image(pdf_path, page, clip_rect)
            b64_image = encode_image_to_base64(image)

            output = invoke_claude_image(prompt_text, b64_image)
            append_to_output(page + 1, output)

        except Exception as e:
            print(f"Failed page {page + 1}: {e}")


# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    try:
        start_page = int(input("Enter start page number (1-based): ").strip())
        end_page = int(input("Enter end page number (1-based): ").strip())
    except Exception:
        print("Invalid input. Exiting.")
        exit(1)

    process_pages(PDF_PATH, start_page, end_page)
    print("Processing complete.")
