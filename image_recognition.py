# Signage/Promotional Image Recognition IP: Refactored for reusable use case, structured output, and modular design
import base64
import json
import logging
import os
import pandas as pd
import requests

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from typing import Optional, List

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
ENDPOINT_BASE = os.getenv('AZURE_OPENAI_ENDPOINT')
MODEL_DEPLOYMENT = os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT')
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
DETAIL_LEVEL = "high"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get Azure AD token provider for Azure OpenAI API
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# Create AzureOpenAI client
client = AzureOpenAI(
  azure_endpoint = ENDPOINT_BASE, 
  azure_ad_token_provider=token_provider,
  api_version="2024-08-01-preview"
)

# Pydantic models for data validation and serialization
class SignageDetection(BaseModel):
    signage_UUID: str  # The promotional signage UUID (e.g., AB12380, XY98765)
    found: bool  # True if the signage was found, False otherwise
    location_description: Optional[str]  # Description of where the signage was found (or None)

class ShelfImageResult(BaseModel):
    shelf_image_name: str  # Name of the shelf image being processed
    findings: List[SignageDetection]  # List of signage detections for this image

# Functions for processing shelf images and getting the results
def create_payload_messages(shelf_image_filename: str, shelf_image: str, signage_images: dict) -> list:
    """
    Create the payload messages for the Azure OpenAI API request.
    """
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are an expert retail auditor tasked with identifying promotional signage in store shelf images. 
                        Your job is to analyze the given store shelf image and determine if any of the provided promotional signage UUIDs are visible in the image.
                        Provide your findings in table format with columns for the promotional signage UUID, whether the signage was found (Yes/No), and a brief description of its location on the shelf.
                        If no signage is detected, mark it as 'No' in the table.
                        Provide your findings in the following table format:
                        Signage UUID	Found (Yes/No)	Location Description"""
                },
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Promotional signage images and their associated UUIDs:"
                },
                *[
                    item 
                    for UUID in signage_images.items()
                    for item in [
                        {
                            "type": "text",
                            "text": f"\nSignage UUID: {UUID}\n" # Add description if needed ", Description: {description}"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{signage_images[UUID]}"
                        }
                    ]
                ],
                {
                    "type": "text",
                    "text": f"\nReview the following store shelf image: {shelf_image_filename}\n"
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{shelf_image}"
                }
            ]
        }
    ]

@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff strategy
    retry=retry_if_exception_type(Exception)  # Retry on these exceptions
)
def get_analysis_request(messages: list) -> dict:
    """
    Send messages to Azure OpenAI API using AzureOpenAI client.
    """
    return client.beta.chat.completions.parse(
        model=MODEL_DEPLOYMENT,
        messages=messages,
        response_format=ShelfImageResult,
        temperature=0.1,
        top_p=0.95,
        max_tokens=800,
    )

def process_single_image(shelf_image_file: str, shelf_image_folder: str, signage_images: dict) -> Optional[dict]:
    """
    Process a single shelf image and collect findings.
    """
    try:
        # Encode the current shelf image to base64
        with open(os.path.join(shelf_image_folder, shelf_image_file), 'rb') as img_file:
            shelf_image = base64.b64encode(img_file.read()).decode('ascii')

        # Create payload for the current shelf image
        payload_messages = create_payload_messages(shelf_image_file, shelf_image, signage_images)
    
        # Send the request and get the result
        completion = get_analysis_request(payload_messages)

        content = completion.choices[0].message.content
        analysis_results = json.loads(content)
    
        # Extract findings
        data = []
        for shelf_image_result in analysis_results.get('shelf_images_results', []):
            if 'findings' in shelf_image_result:
                findings = [SignageDetection(**finding) for finding in shelf_image_result['findings']]
                for finding in findings:
                    data.append({
                        "Shelf Image Name": shelf_image_result.get('shelf_image_name', 'unknown'),
                        "Signage UUID": finding.signage_UUID,
                        "Found": finding.found,
                        "Location Description": finding.location_description
                    })
            else:
                logging.warning(f"'findings' key not found in shelf image result for {shelf_image_result.get('shelf_image_name', 'unknown')}")
        return data

    except Exception as e:
        logging.error(f"Error processing image {shelf_image_file}: {e}")
        return None

def main():
    # Get all signage image files in the folder
    signage_image_folder = "images/signage"
    signage_image_files = [f for f in os.listdir(signage_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Encode each signage image and store in a dictionary with the filename as the key
    signage_images = {file: base64.b64encode(open(os.path.join(signage_image_folder, file), 'rb').read()).decode('ascii') for file in signage_image_files}

    # Prepare the DataFrame
    df_columns = ["Shelf Image Name", "Signage UUID", "Found", "Location Description"]
    result_csv_file = 'shelf_analysis_results.csv'

    # Create CSV with headers
    pd.DataFrame(columns=df_columns).to_csv(result_csv_file, index=False)

    # Process all shelf images
    shelf_image_folder = "images/shelf"
    shelf_image_files = [f for f in os.listdir(shelf_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for shelf_image_file in tqdm(shelf_image_files, desc="Processing Shelf Images"):
        data = process_single_image(shelf_image_file, shelf_image_folder, signage_images)
        if data:
            # Write partial results to CSV
            partial_df = pd.DataFrame(data, columns=df_columns)
            partial_df.to_csv(result_csv_file, mode='a', header=False, index=False)

    # Load the final DataFrame and provide a summary
    final_df = pd.read_csv(result_csv_file)
    summary = final_df.groupby(['Signage UUID', 'Found']).size().reset_index(name='Count')
    true_summary = final_df[final_df['Found'] == True].groupby('Signage UUID')['Shelf Image Name'].apply(list).reset_index(name='Images Found')
    accuracy = (final_df['Found'].sum() / len(final_df)) * 100
    logging.info(f"Summary of Findings:\n{summary}")
    logging.info(f"Images with True Findings:\n{true_summary.to_string(index=False)}")
    logging.info(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()