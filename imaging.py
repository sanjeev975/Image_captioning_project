from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

# Load the pretrained model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate multiple captions
def generate_multiple_captions(image_path, num_captions=3):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate captions
    captions = []
    for _ in range(num_captions):
        output_ids = model.generate(
            pixel_values,
            max_length=16,
            num_beams=5,
            num_return_sequences=1,  # Return one sequence per iteration
            do_sample=True,         # Enable sampling
            top_k=50,               # Top-k sampling
            temperature=1.0         # Adjust temperature for diversity
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)

    return captions

# Example usage
image_path = "/Users/sa/python/lop.jpg"  # Replace with your image file path
captions = generate_multiple_captions(image_path, num_captions=5)
print("Generated Captions:")
for i, caption in enumerate(captions):
    print(f"{i + 1}: {caption}")
