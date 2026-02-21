#!/usr/bin/env python3
"""
Interactive PDF reviewer:
- Raw mode (selectable text only): lets you edit margins AND edit the extracted text before saving.
- Flowchart mode (vision LLM): sends the clipped-region image to Claude 3.5 Sonnet for summarization.
"""

import base64
import os
import sys
import json
import tempfile
import subprocess
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import boto3

# ------------------------ CONFIG ------------------------
PDF_PATH = "all.pdf"
OUTPUT_PATH = "nccn_trial.txt"
AWS_REGION = "us-east-1"

# Model you want to use (doesn't require inference profile)
MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"
# --------------------------------------------------------

# ------------------ Helpers ------------------

def extract_text_with_margins(pdf_path, page_num, margin_left, margin_top, margin_right, margin_bottom):
    """Return tuple (text, clip_rect) where text is selectable text inside the clipped rectangle."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    rect = page.rect
    clipped_rect = fitz.Rect(
        rect.x0 + margin_left,
        rect.y0 + margin_top,
        rect.x1 - margin_right,
        rect.y1 - margin_bottom
    )
    text = page.get_text("text", clip=clipped_rect)
    doc.close()
    return text, clipped_rect

def render_clipped_image(pdf_path, page_num, clip_rect):
    """Render full page then crop to the clip_rect and return a PIL image."""
    page_img = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)[0]
    # compute dpi scale: pixels per PDF unit
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
    cropped = page_img.crop(crop_box)
    return cropped

def encode_image_to_base64(pil_img):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def invoke_claude_image(prompt_text, b64_image):
    """Call Bedrock runtime with modelId (anthropic.claude-3-5-sonnet)."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{
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
        }],
        "max_tokens": 4096
    }

    resp = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=json.dumps(payload)
    )

    raw = resp["body"].read().decode("utf-8")
    data = json.loads(raw)
    return data.get("content", [{}])[0].get("text", "")

def launch_editor(initial_text):
    """Open the user's $EDITOR (or nano) with the initial_text and return edited content."""
    editor = os.environ.get("EDITOR", "nano")
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tf_name = tf.name
    try:
        tf.write(initial_text.encode("utf-8"))
    finally:
        tf.close()

    subprocess.call([editor, tf_name])

    with open(tf_name, "r", encoding="utf-8") as f:
        final_text = f.read()

    try:
        os.unlink(tf_name)
    except Exception:
        pass
    return final_text

def append_to_output(page_num, content):
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== PAGE {page_num} ===\n")
        f.write(content.strip() + "\n")
        f.write("=== END PAGE ===\n")


# ------------------ Interactive Loop ------------------

def interactive_run(pdf_path, start_page):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Loaded '{pdf_path}' with {total_pages} pages.")
    page = max(1, start_page) - 1  # zero-indexed

    while page < total_pages:
        print("\n" + "="*70)
        print(f"Page {page+1}/{total_pages}")
        print("Actions: (s)kip  (r)aw text  (f)lowchart->paragraph  (q)uit")
        action = input("Choose action (s/r/f/q): ").strip().lower()

        if action == "q":
            print("Quitting.")
            break
        if action == "s":
            page += 1
            continue

        # --------- RAW MODE (selectable text) ----------
        if action == "r":
            # default margins (units are PDF points: 1 pt = 1/72 inch)
            margin_left = 0
            margin_right = 0
            margin_top = 80
            margin_bottom = 30

            while True:
                text, clip_rect = extract_text_with_margins(pdf_path, page, margin_left, margin_top, margin_right, margin_bottom)

                print("\n--- Extracted Text (PDF selectable text only) ---\n")
                print(text if text.strip() else "[NO TEXT FOUND WITH CURRENT MARGINS]")

                print("\nOptions: (a)ccept/save  (m)odify margins  (e)dit text  (re)extract  (s)kip page  (q)quit raw mode")
                choice = input("Choice: ").strip().lower()

                if choice == "a":
                    append_to_output(page+1, text)
                    print("Saved extracted text.")
                    break

                elif choice == "m":
                    try:
                        new_top = input(f"Top margin (current {margin_top}) - press Enter to keep: ").strip()
                        new_bottom = input(f"Bottom margin (current {margin_bottom}) - press Enter to keep: ").strip()
                        new_left = input(f"Left margin (current {margin_left}) - press Enter to keep: ").strip()
                        new_right = input(f"Right margin (current {margin_right}) - press Enter to keep: ").strip()
                        if new_top != "":
                            margin_top = int(new_top)
                        if new_bottom != "":
                            margin_bottom = int(new_bottom)
                        if new_left != "":
                            margin_left = int(new_left)
                        if new_right != "":
                            margin_right = int(new_right)
                    except ValueError:
                        print("Invalid input — margins must be integers. Try again.")
                    continue  # re-extract with new margins

                elif choice == "e":
                    # open editor with current extracted text so user can modify content before saving
                    edited = launch_editor(text)
                    print("\n--- Edited Text Preview ---\n")
                    print(edited if edited.strip() else "[EMPTY AFTER EDIT]")
                    if input("Accept edited text and save? (y/n): ").strip().lower() == "y":
                        append_to_output(page+1, edited)
                        print("Saved edited text.")
                        break
                    else:
                        print("Edited text not saved. Returning to raw options.")
                        continue

                elif choice == "re":
                    # just re-extract (loop will display text again)
                    continue

                elif choice == "s":
                    print("Skipping page (raw mode).")
                    break

                elif choice == "q":
                    print("Exiting raw mode to main menu.")
                    return

                else:
                    print("Unknown option. Please choose again.")
                    continue

            page += 1
            continue

        # --------- FLOWCHART MODE (LLM on clipped image) ----------
        if action == "f":
            # use default margins to compute clip region (so headers/footers are removed)
            margin_left = 0
            margin_right = 0
            margin_top = 60
            margin_bottom = 30

            # extract only to get clip_rect (we don't rely on extracted selectable text here)
            _, clip_rect = extract_text_with_margins(pdf_path, page, margin_left, margin_top, margin_right, margin_bottom)

            # render clipped image and send to LLM
            clipped_img = render_clipped_image(pdf_path, page, clip_rect)
            b64_img = encode_image_to_base64(clipped_img)

            prompt_text = (
                "You are given an image of a PDF page that may contain a flowchart or a table or diagram. convert this page of guidelines that has flowchart into paragraph that can be used for RAG"
                "Convert the visual guidance into a single clear, concise paragraph suitable for RAG ingestion. "
                "Do NOT add facts not present in the image or invent steps."
		"if its a paragraph retain it as is and remove the superscript only"
		"if its a table, covert it into json format and give the output"
		"if and when any drugs are mentioned based on gene deletion or mutation like EGFR exon, mention them in the paragraph"
		"Include the blue references mentioned in brackets on the right side of flowchart in the output paragraph"
		"Do not include the lines - Note: All recommendations are category 2A unless otherwise indicated,  mentioned at the bottom"
		" Do not include the word - References mentioned at the bottom right corner"
		"Include the page number given in the bottom right of the pdf thats a mix of letters and numbers in a new line below output text"
            )	

            print("Sending clipped image to LLM...")
            try:
                output = invoke_claude_image(prompt_text, b64_img)
            except Exception as e:
                print(f"LLM invocation failed: {e}")
                output = ""

            print("\n--- LLM Output ---\n")
            print(output if output.strip() else "[NO OUTPUT FROM LLM]")

            while True:
                post = input("\n(a)ccept/save  (e)dit  (re)retry  (s)kip page  (q)quit: ").strip().lower()
                if post == "a":
                    append_to_output(page+1, output)
                    print("Saved LLM paragraph.")
                    break
                elif post == "e":
                    edited = launch_editor(output)
                    print("\n--- Edited LLM Output Preview ---\n")
                    print(edited if edited.strip() else "[EMPTY AFTER EDIT]")
                    if input("Accept edited and save? (y/n): ").strip().lower() == "y":
                        append_to_output(page+1, edited)
                        print("Saved edited LLM paragraph.")
                        break
                    else:
                        print("Edited text not saved. Returning to LLM options.")
                        continue
                elif post == "re":
                    print("Retrying LLM...")
                    try:
                        output = invoke_claude_image(prompt_text, b64_img)
                    except Exception as e:
                        print(f"LLM invocation failed: {e}")
                        output = ""
                    print("\n--- New LLM Output ---\n")
                    print(output if output.strip() else "[NO OUTPUT FROM LLM]")
                    continue
                elif post == "s":
                    print("Skipping page (flowchart mode).")
                    break
                elif post == "q":
                    print("Exiting to main menu.")
                    return
                else:
                    print("Unknown option.")
                    continue

            page += 1
            continue

        # If reached here, unknown action
        print("Unknown action. Please choose again.")

# ------------------ Entry point ------------------

if __name__ == "__main__":
    try:
        start_page = int(input("Enter start page number (1-based): ").strip())
    except Exception:
        print("Invalid page. Starting from page 1.")
        start_page = 1
    interactive_run(PDF_PATH, start_page)


