import re
import os
import json
import torch
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ----------------------------
# Config
# ----------------------------

demo_data = [
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000070.jpg',
        'width': 640,
        'height': 480,
        'left': 52,
        'top': 372,
        'right': 636,
        'bottom': 443,
        'question': (
            "What equipment is used for snowboarding? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000092.jpg',
        'width': 640,
        'height': 427,
        'left': 417,
        'top': 120,
        'right': 583,
        'bottom': 426,
        'question': (
            "What do you use to eat a cake? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000141.jpg',
        'width': 640,
        'height': 399,
        'left': 3,
        'top': 33,
        'right': 110,
        'bottom': 216,
        'question': (
            "Where do we control the water from? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000257.jpg',
        'width': 640,
        'height': 480,
        'left': 101,
        'top': 415,
        'right': 176,
        'bottom': 479,
        'question': (
            "What is used for carrying a baby? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000335.jpg',
        'width': 640,
        'height': 480,
        'left': 270,
        'top': 16,
        'right': 345,
        'bottom': 86,
        'question': (
            "What do we use to watch movies? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type": "FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000410.jpg',
        'width': 640,
        'height': 480,
        'left': 38,
        'top': 103,
        'right': 303,
        'bottom': 362,
        'question': (
            "What can cut paper easily? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000412.jpg',
        'width': 640,
        'height': 426,
        'left': 471,
        'top': 162,
        'right': 638,
        'bottom': 423,
        'question': (
            "What living creature can speak? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000436.jpg',
        'width': 427,
        'height': 640,
        'left': 221,
        'top': 237,
        'right': 321,
        'bottom': 310,
        'question': (
            "What do we eat for dessert? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000459.jpg',
        'width': 516,
        'height': 640,
        'left': 372,
        'top': 245,
        'right': 475,
        'bottom': 316,
        'question': (
            "What can be used to take pictures? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
    {
        'image': 'https://toloka-cdn.azureedge.net/wsdmcup2023/000000000492.jpg',
        'width': 640,
        'height': 383,
        'left': 166,
        'top': 248,
        'right': 281,
        'bottom': 353,
        'question': (
            "What do parents love and care for? "
            "Output the bbox coordinates of what is asked in the question. "
            'The bbox coordinates shold be in the following format: [{"_type":"FunctionCall", "arguments": "{"bbox_2d": [x1, y1, x2, y2], "label": "label_of_bbox"}", "name": "image_bbox_tool", "_class_name": "FunctionCall", "_bases": ["BaseModel"]}]'
        )
    },
]
output_dir = "/users/jsaydali/scratch/run_outputs/debug_output"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Load Model & Processor
# ----------------------------
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
device = model.device

# ----------------------------
# Utilities
# ----------------------------
def load_image(image_url):
    """Download and return RGB PIL image."""
    return Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

def run_inference(image_url, question):
    """Run inference using chat template + qwen-vl-utils."""
    # Load original
    original_image = load_image(image_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": original_image},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Prepare text prompt
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Use qwen-vl-utils for vision preprocessing (resizes internally)
    image_inputs, video_inputs = process_vision_info(messages)

    # At this point, image_inputs is a list of resized PIL.Image(s)
    resized_image = image_inputs[0]

    # Tokenize with processor
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to(device)

    # Generate
    output_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return original_image, resized_image, output_text

def extract_bbox(model_output):
    """
    Extract and parse the bbox JSON from model output.
    The bbox format is expected as:
    {"bbox_2d": [x1, y1, x2, y2], "label": "label"}
    """
    # Regex to find JSON object with bbox_2d inside
    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}'
    match = re.search(pattern, model_output)
    if not match:
        return None
    
    json_str = match.group(0)
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "bbox_2d" in parsed:
            return parsed
    except json.JSONDecodeError:
        return None
    
    return None

def draw_bbox(image, bbox_json, save_path):
    """Draw bbox on image."""
    if bbox_json is None:
        image.save(save_path)
        return
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox_json["bbox_2d"]
    label = bbox_json.get("label", "object")
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, max(0, y1 - 12)), label, fill="red")
    image.save(save_path)

# ----------------------------
# Main Loop
# ----------------------------
for idx, sample in tqdm(enumerate(demo_data), desc="Processing samples", total=len(demo_data)):
    image_url = sample["image"]
    question = sample["question"]

    # Run inference (resizing happens inside)
    original_image, resized_image, model_output = run_inference(image_url, question)
    original_size = original_image.size
    resized_size = resized_image.size

    # Extract bbox
    bbox_json = extract_bbox(model_output)

    # Create per-sample directory
    sample_dir = os.path.join(output_dir, f"sample_{idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # Save text output
    with open(os.path.join(sample_dir, "output.txt"), "w") as f:
        f.write("Question: " + question + "\n\n")
        f.write("Original Image Size: " + str(original_size) + "\n")
        f.write("Resized Image Size: " + str(resized_size) + "\n\n")
        f.write("Model Output:\n" + model_output + "\n\n")
        f.write("Extracted BBox:\n" + json.dumps(bbox_json, indent=2) + "\n")

    # Save images
    original_image.save(os.path.join(sample_dir, "input_original.jpg"))
    resized_image.save(os.path.join(sample_dir, "input_resized.jpg"))
    draw_bbox(resized_image.copy(), bbox_json, os.path.join(sample_dir, "bbox.jpg"))

print("✅ Done. Results in:", output_dir)