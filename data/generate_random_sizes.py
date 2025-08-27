import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datasets import Dataset
import argparse

# Configuration
FONT_PATH = "arial.ttf"
IMG_SIZE = 1024

# Variations
SIZES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
OUT_OF_BOUND_SIZES = [4, 5, 6, 7] # out-of-bound sizes for `Qwen/Qwen2.5-VL-3B-Instruct`

# Default values
DEFAULT_POSITION = "center"
DEFAULT_OPACITY = 255

# 20 Light background colors (RGB values)
LIGHT_BACKGROUNDS = [
    (255, 255, 255),  # White
    (248, 248, 255),  # Ghost White
    (245, 245, 220),  # Beige
    (255, 228, 225),  # Misty Rose
    (240, 248, 255),  # Alice Blue
    (255, 250, 240),  # Floral White
    (253, 245, 230),  # Old Lace
    (255, 239, 213),  # Papaya Whip
    (255, 228, 196),  # Bisque
    (255, 218, 185),  # Peach Puff
    (250, 235, 215),  # Antique White
    (255, 240, 245),  # Lavender Blush
    (240, 255, 240),  # Honeydew
    (255, 255, 240),  # Ivory
    (240, 255, 255),  # Azure
    (245, 255, 250),  # Mint Cream
    (255, 245, 238),  # Seashell
    (250, 240, 230),  # Linen
    (255, 248, 220),  # Cornsilk
    (255, 255, 224),  # Light Yellow
]

# 20 Dark text colors (RGB values)
DARK_TEXT_COLORS = [
    (0, 0, 0),        # Black
    (25, 25, 112),    # Midnight Blue
    (139, 0, 0),      # Dark Red
    (0, 100, 0),      # Dark Green
    (72, 61, 139),    # Dark Slate Blue
    (128, 0, 128),    # Purple
    (85, 107, 47),    # Dark Olive Green
    (184, 134, 11),   # Dark Goldenrod
    (165, 42, 42),    # Brown
    (47, 79, 79),     # Dark Slate Gray
    (105, 105, 105),  # Dim Gray
    (75, 0, 130),     # Indigo
    (128, 0, 0),      # Maroon
    (0, 0, 139),      # Dark Blue
    (0, 128, 0),      # Green
    (139, 69, 19),    # Saddle Brown
    (148, 0, 211),    # Dark Violet
    (220, 20, 60),    # Crimson
    (255, 140, 0),    # Dark Orange
    (138, 43, 226),   # Blue Violet
]

# 100 Intentionally misspelled words
MISSPELLED_WORDS = [
    "restaraunt", "definitly", "seperate", "occured", "recieve", "neccessary", "accomodate", "acheive", "beleive", "wierd",
    "freind", "knowlege", "tomorow", "occassion", "begining", "existance", "diffrent", "independant", "enviroment", "goverment",
    "intresting", "rythm", "publically", "exagerate", "mispell", "pronounciation", "noticable", "maintainance", "embarass", "harrass",
    "occurance", "perseverence", "posession", "priviledge", "reccomend", "transfered", "untill", "neice", "athiest", "bizzare",
    "cemetary", "concious", "dissapoint", "ecstacy", "foriegn", "gurantee", "hieght", "ignorence", "jeapordy", "liesure",
    "millenium", "neccessity", "occured", "persue", "questionaire", "resteraunt", "seperate", "truely", "usefull", "vaccum",
    "wellcome", "excercise", "yeild", "zephyr", "absense", "boundry", "cemetry", "dicipline", "effeciency", "fourty",
    "garantee", "hankerchief", "ilegal", "jewelery", "knowlege", "liason", "memento", "noticable", "occured", "paralell",
    "privilage", "questionaire", "refered", "succesful", "tommorrow", "unecessary", "vengence", "wensday", "xerox", "yeilds",
    "zephyre", "accomodation", "begining", "calender", "definately", "embarasing", "freind", "goverment", "heigth", "independant"
]

# Utility functions
def get_safe_position(font, text, position_type="center"):
    """Get a safe position for text within image bounds."""
    bbox = font.getbbox(text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if position_type == "random":
        x = random.randint(0, max(0, IMG_SIZE - w))
        y = random.randint(0, max(0, IMG_SIZE - h))
    else:  # center
        x = (IMG_SIZE - w) // 2
        y = (IMG_SIZE - h) // 2
    return (x, y)

def create_text_image(text, background_color, text_color, font_size):
    """Create an image with text using specified parameters."""
    # Start with base image
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), background_color)
    
    # Create text layer
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Get safe position
    pos = get_safe_position(font, text, DEFAULT_POSITION)
    
    # Create a transparent overlay for text
    txt_img = Image.new('RGBA', (IMG_SIZE, IMG_SIZE), (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)
    
    # Draw text with specified color and opacity
    text_color_with_alpha = (*text_color, DEFAULT_OPACITY)
    draw.text(pos, text, font=font, fill=text_color_with_alpha)
        
    # Composite text onto image
    img = img.convert('RGBA')
    final_img = Image.alpha_composite(img, txt_img)
    final_img = final_img.convert('RGB')
    
    return final_img

def save_image_and_text(img, filename, text, output_dir):
    """Save image and record parameters in dataframe."""
    global df
    img_path = os.path.join(output_dir, filename)
    img.save(img_path)
    
    # Add to dataframe with HuggingFace format
    img_abs_path = os.path.abspath(img_path)
    new_row = pd.DataFrame([{
        'problem': "What is the text written in the image?",
        'answer': text,
        'images': [img_abs_path] 
    }])
    df = pd.concat([df, new_row], ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/sizes")
    parser.add_argument("--oob_ratio", type=float, default=0.9)
    parser.add_argument("--num_images", type=int, default=10000)
    args = parser.parse_args()

    # parse arguments
    output_dir = args.output_dir
    oob_ratio = args.oob_ratio
    num_images = args.num_images

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(columns=['problem', 'answer', 'images'])

    # Generate images
    print(f"Generating {num_images} varied images...")

    for i in tqdm(range(num_images), desc="Generating images", total=num_images):
        # Randomly select parameters for each image
        background_color = random.choice(LIGHT_BACKGROUNDS)
        text_color = random.choice(DARK_TEXT_COLORS)
        text = random.choice(MISSPELLED_WORDS)
        # sample with 90% for out-of-bound sizes
        weights = [oob_ratio / len(OUT_OF_BOUND_SIZES) if angle in OUT_OF_BOUND_SIZES else (1 - oob_ratio) / (len(SIZES) - len(OUT_OF_BOUND_SIZES)) for angle in SIZES]
        font_size = random.choices(SIZES, weights=weights, k=1)[0]
        
        # Create the image
        img = create_text_image(
            text=text,
            background_color=background_color,
            text_color=text_color,
            font_size=font_size
        )
        
        # Save with descriptive filename
        filename = f"varied_text_{i+1:04d}.jpg"
        save_image_and_text(img, filename, text, output_dir)

    # Create HuggingFace Dataset
    print("Creating HuggingFace Dataset...")
    hf_dataset = Dataset.from_pandas(df)

    # Save the HuggingFace dataset
    dataset_path = os.path.join(output_dir, "text_recognition_dataset")
    hf_dataset.save_to_disk(dataset_path)

    # Also save as CSV for reference
    df.to_csv(os.path.join(output_dir, "dataset_metadata.csv"), index=False)

    print(f"\nGeneration complete!")
    print(f"Total images generated: {len(df)}")
    print(f"HuggingFace dataset saved to: {dataset_path}")
    print(f"Metadata CSV saved as: {os.path.join(output_dir, 'dataset_metadata.csv')}")

    # Display statistics
    print("\nDataset Statistics:")
    print(f"Background colors used: {len(LIGHT_BACKGROUNDS)}")
    print(f"Text colors used: {len(DARK_TEXT_COLORS)}")
    print(f"Misspelled words used: {len(MISSPELLED_WORDS)}")
    print(f"Total combinations possible: {len(LIGHT_BACKGROUNDS)} × {len(DARK_TEXT_COLORS)} × {len(MISSPELLED_WORDS)} = {len(LIGHT_BACKGROUNDS) * len(DARK_TEXT_COLORS) * len(MISSPELLED_WORDS):,}")

    print("\nDataset Schema:")
    print(hf_dataset)

    print("\nFirst 5 examples:")
    for i in range(min(5, len(hf_dataset))):
        example = hf_dataset[i]
        print(f"Example {i+1}:")
        print(f"  Problem: {example['problem']}")
        print(f"  Answer: {example['answer']}")
        print(f"  Image path: {example['images']}")
        print()

    print(f"\nDataset shape: {len(hf_dataset)} rows, {len(hf_dataset.column_names)} columns")
    print(f"Column names: {hf_dataset.column_names}")

    # Show distribution of answers
    print("\nAnswer distribution:")
    answer_counts = pd.Series([example['answer'] for example in hf_dataset]).value_counts()
    print(f"Unique answers: {len(answer_counts)}")
    print("Top 10 most frequent answers:")
    print(answer_counts.head(10))