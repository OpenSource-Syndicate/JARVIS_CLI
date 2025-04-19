from google import genai
from groq import Groq
import base64
import os
from ..database.data import ConversationDB

def process_img_gemini(image_path, prompt, api_key="GOOGLE_API_KEY"):
    client = genai.Client(api_key=api_key)
    db = ConversationDB()
    
    # Upload the image file
    myfile = client.files.upload(file=image_path)
    
    # Generate caption
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[myfile, prompt])
    
    # Store the processed data in database
    result_text = response.text if response.text else "No response generated"
    db.save_message(
        context=f"image_processing:{image_path}",
        sender="assistant",
        message=result_text,
        timestamp=datetime.now().isoformat()
    )
    db.save_message(
        context=f"image_processing:{image_path}",
        sender="system",
        message=f"Prompt: {prompt}\nResponse: {result_text}"
    )
    
    return result_text

def process_img_groq(image_path, prompt="What's in this image?", api_key=None):
    # Function to encode the image
    def encode_image(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Initialize Groq client
    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
    
    # Get base64 encoded image
    base64_image = encode_image(image_path)
    
    # Make API call
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    return chat_completion.choices[0].message.content