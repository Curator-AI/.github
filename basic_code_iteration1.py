import openai
import torch
from PIL import Image
import clip
import requests
import io
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# The path to your audio file
audio_file_path = '/Users/ns/Downloads/audio_1.mp3'  

# Open the audio file
with open(audio_file_path, 'rb') as audio_file:
    # Make the request to the Whisper API to transcribe the audio
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_file,
    )

# Print the transcribed text
print("Transcription:", response['text'])

# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to generate a caption for the input image
def generate_caption(image_path):
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Use a set of pre-defined candidate descriptions (you can make these dynamic or custom)
    descriptions = [
        "A room with a wooden table",
        "A modern chair in a room setting",
        "A living room with furniture",
        "A cozy indoor space",
        "A wooden table in a simple room"
    ]
    text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(device)

    # Predict which description best matches the image
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        logits_per_image, logits_per_text = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Select the description with the highest probability
    best_match_idx = probs.argmax()
    return descriptions[best_match_idx]

# Function to generate an image based on a prompt
def generate_image_based_on_prompt(text_prompt):
    openai.api_key = os.getenv('OPENAI_API_KEY')
 # Use environment variables or secure key management

    # Send a request to OpenAI to generate a new image based on a text prompt
    response = openai.Image.create(
        prompt=text_prompt,
        n=1,
        size="1024x1024"
    )

    # Get the URL of the generated image
    generated_image_url = response['data'][0]['url']
    
    # Download and show/save the generated image
    generated_image_response = requests.get(generated_image_url)
    generated_image = Image.open(io.BytesIO(generated_image_response.content))
    generated_image.show()
    generated_image.save('generated_image.png')
    print(f"Generated image saved as 'generated_image.png'. URL: {generated_image_url}")

    return generated_image_url

# Example usage
image_path = '/Users/ns/Downloads/table_in_room.jpg'  # Path to your input image
text_prompt = "I want a chair appropriate for this setting"  # Your custom prompt

# Step 1: Generate a caption/description from the image
caption = generate_caption(image_path)
print(f"Generated Caption: {caption}")

# Step 2: Combine the image-derived caption with the user's prompt
combined_prompt = f"{caption}. {text_prompt}"
print(f"Combined Prompt: {combined_prompt}")

# Step 3: Generate a new image based on the combined prompt
generated_image_url = generate_image_based_on_prompt(combined_prompt)


# Sending the image to Google Lens

api_key = os.getenv('SERPAPI_API_KEY')
# URL to the SerpApi Google Images endpoint
SERPAPI_ENDPOINT = 'https://serpapi.com/search.json'

# The URL of the image you want to search
image_url = generated_image_url 

# Define the parameters for the search
params = {
    "engine": "google_lens",
    "url": image_url,
    "api_key": api_key
}

# Make the request to SerpApi
response = requests.get(SERPAPI_ENDPOINT, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Assuming the response includes shopping/product links
    for product in data.get("shopping_results", []):
        print("Product Name:", product.get("title"))
        print("Product Link:", product.get("link"))
        print("Price:", product.get("price"))
        print("-----------")
else:
    print("Error:", response.status_code, response.text)
