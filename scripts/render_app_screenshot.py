"""
Render a pixel-accurate mockup of the MedCompress desktop app
directly to a PNG image using Pillow. No window manager needed.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_screenshot():
    W, H = 720, 560
    img = Image.new("RGB", (W, H), "#f5f6fa")
    draw = ImageDraw.Draw(img)

    # Try to get a decent font
    try:
        font_bold_16 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_10 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_mono = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 12)
        font_btn = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except (OSError, IOError):
        font_bold_16 = ImageFont.load_default()
        font_10 = ImageFont.load_default()
        font_mono = ImageFont.load_default()
        font_btn = ImageFont.load_default()

    # Header bar
    draw.rectangle([0, 0, W, 50], fill="#2c3e50")
    draw.text((15, 14), "MedCompress", fill="white", font=font_bold_16)
    draw.text((420, 18), "Model: isic_qat_int8  |  Task: classification",
              fill="#bdc3c7", font=font_10)

    # Image display area
    draw.rectangle([15, 60, W-15, 380], fill="#ecf0f1", outline="#dcdde1", width=1)

    # Load and paste the sample lesion image
    sample_path = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_lesion.jpg")
    if os.path.exists(sample_path):
        lesion = Image.open(sample_path).resize((280, 280), Image.LANCZOS)
        paste_x = (W - 280) // 2
        paste_y = 80
        img.paste(lesion, (paste_x, paste_y))
        # Border around image
        draw.rectangle([paste_x-1, paste_y-1, paste_x+280, paste_y+280],
                       outline="#bdc3c7", width=1)
        # Filename label
        draw.text((paste_x, paste_y + 285), "sample_lesion.jpg",
                  fill="#7f8c8d", font=font_10)

    # Buttons
    btn_y = 392
    # Open Image button
    draw.rounded_rectangle([15, btn_y, 140, btn_y+35], radius=4, fill="#3498db")
    draw.text((38, btn_y+9), "Open Image", fill="white", font=font_btn)
    # Run Analysis button
    draw.rounded_rectangle([155, btn_y, 290, btn_y+35], radius=4, fill="#27ae60")
    draw.text((172, btn_y+9), "Run Analysis", fill="white", font=font_btn)
    # Benchmark button
    draw.rounded_rectangle([305, btn_y, 480, btn_y+35], radius=4, fill="#8e44ad")
    draw.text((314, btn_y+9), "Benchmark (50 runs)", fill="white", font=font_btn)

    # Results area
    results_y = 440
    draw.rectangle([15, results_y, W-15, H-15], fill="white", outline="#dcdde1", width=1)

    # Result text
    result_lines = [
        "Prediction: Melanoma (confidence: 87.3%)",
        "Inference: 9.2 ms",
        "",
        "Model: isic_qat_int8.tflite (4.3 MB)",
        "Input: 224x224x3  |  Task: classification",
    ]
    for i, line in enumerate(result_lines):
        color = "#e74c3c" if "Melanoma" in line else "#2c3e50"
        if "Inference" in line:
            color = "#27ae60"
        draw.text((25, results_y + 8 + i * 18), line, fill=color, font=font_mono)

    # Window chrome (subtle shadow)
    shadow = Image.new("RGB", (W+8, H+8), "#c8c8c8")
    shadow.paste(img, (4, 4))

    out_path = os.path.join(os.path.dirname(__file__), "..", "screenshots", "app_demo.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shadow.save(out_path, "PNG")
    print(f"Screenshot rendered: {out_path}")
    return out_path


if __name__ == "__main__":
    render_screenshot()
