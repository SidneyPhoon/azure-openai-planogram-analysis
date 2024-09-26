#script to generate SOM image from input image

import requests
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()

SOM_Image_Generation_URL = os.getenv('SOM_Image_Generation_URL')


input_image_path = 'images/shelf.jpg'
output_image_path = 'images/shelf_som_output.jpg'
with open(input_image_path, 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

url = SOM_Image_Generation_URL

headers = {'Content-Type': 'application/json'}
data = json.dumps({'image': base64_image})

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    response_data = response.json()
    image_base64 = response_data.get('image')
    if image_base64:
        # print('Base64 Image:', image_base64)
                # Save the base64 image to a file
        with open(output_image_path, 'wb') as output_file:
            output_file.write(base64.b64decode(image_base64))
    else:
        print('No image returned in response')
else:
    print('Error:', response.json())