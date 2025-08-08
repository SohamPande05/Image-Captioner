import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, BlipForConditionalGeneration
from bs4 import BeautifulSoup, Tag

# Load BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL to scrape
url = "https://en.wikipedia.org/wiki/India"

# Fetch and parse the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all <img> tags
img_elements = soup.find_all('img')

with open("captions.txt", "w", encoding="utf-8") as caption_file:

    for img_element in img_elements:

        # Ensure img_element is a Tag instance
        if not isinstance(img_element, Tag):
            continue

        img_url = img_element.get('src')
        if not isinstance(img_url, str):
            continue  # Skip if src is None or not a string

        # Skip SVGs or invisible images
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Fix relative or protocol-relative URLs
        if img_url.startswith('//'):
            img_url = "https:" + img_url

        elif img_url.startswith('/'):
            img_url = "https://en.wikipedia.org" + img_url
        
        # Skip invalid URLs
        elif not img_url.startswith('http'):
            continue  

        try:
            # Download the image
            img_response = requests.get(img_url, timeout=10)
            content_type = img_response.headers.get('Content-Type', '')

            if 'image' not in content_type:
                print(f"❌ Skipped non-image content: {content_type}")
                continue

            # Load and check image size
            raw_image = Image.open(BytesIO(img_response.content)).convert('RGB')

            if raw_image.size[0] * raw_image.size[1] < 400:
                continue  # Skip tiny images

            # Process with BLIP
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Save the caption
            caption_file.write(f"{img_url}: {caption}\n")
            print(f"✅ Captioned: {img_url}")

        except Exception as e:
            print(f"❌ Error processing {img_url}: {e}")
            continue