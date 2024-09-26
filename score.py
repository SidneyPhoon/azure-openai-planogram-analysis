# script to generate SOM image from input image.  This is created from the sample code in the SOM repo

import base64
import io
from PIL import Image 
import requests 
import os
import base64

from pathlib import Path
import json

import torch
import argparse

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np

from flask import Flask, request, jsonify


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    _image = image.convert('RGB')
    _mask = image.convert('L') #if image else None

    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(_mask))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        return output

'''
launch app
'''

# demo = gr.Blocks()
# image = gr.ImageMask(label="Input", type="pil", sources=["upload"], interactive=True, brush=gr.Brush(colors=["#FFFFFF"]))
# slider = gr.Slider(1, 3, value=2, label="Granularity", info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]")
# mode = gr.Radio(['Automatic', 'Interactive', ], value='Automatic', label="Segmentation Mode")
# image_out = gr.Image(label="Auto generation",type="pil")
# runBtn = gr.Button("Run")
# slider_alpha = gr.Slider(0, 1, value=0.1, label="Mask Alpha", info="Choose in [0, 1]")
# label_mode = gr.Radio(['Number', 'Alphabet'], value='Number', label="Mark Mode")
# anno_mode = gr.CheckboxGroup(choices=["Mask", "Box", "Mark"], value=['Mask', 'Mark'], label="Annotation Mode")

def init():
    print("init")
    '''
    build args
    '''
    semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
    seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

    semsam_ckpt = "./swinl_only_sam_many2many.pth"
    sam_ckpt = "./sam_vit_h_4b8939.pth"
    seem_ckpt = "./seem_focall_v1.pt"
    print("load configs")
    opt_semsam = load_opt_from_config_file(semsam_cfg)
    opt_seem = load_opt_from_config_file(seem_cfg)
    opt_seem = init_distributed_seem(opt_seem)


    '''
    build model
    '''
    print("load models")
    global model_semsam
    model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
    global model_sam
    model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
    global model_seem
    model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)    




def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

app = Flask(__name__)

@app.route('/image', methods=['POST'])
def process_image():
    data = request.json
    base64_str = data.get('image')

    if not base64_str:
        return jsonify({'error': 'No image provided'}), 400

    slider = data.get('slider', 2)
    mode = data.get('mode', 'Automatic')
    slider_alpha = data.get('slider_alpha', 0.1)
    label_mode = data.get('label_mode', 'Number')
    anno_mode = data.get('anno_mode', ['Mask', 'Mark'])
    try:
        image = base64_to_image(base64_str)
        # image_array = np.array(image)
        image_out = inference(image, slider, mode, slider_alpha, label_mode, anno_mode)
        image_base64 = image_to_base64(Image.fromarray(image_out))
        return jsonify({'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init() 
    app.run(debug=True)


# if __name__=="__main__": 
#     init() 

#         # Load image
#     img_url= 'https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg'
#     image = Image.open(requests.get(img_url, stream=True))
#     # image = Image.open(r"examples/ironing_man.jpg") 

#     slider =2
#     mode = 'Automatic'
    
#     slider_alpha = 0.1
#     label_mode ='Number'
#     anno_mode = ['Mask', 'Mark']

#     image_out = inference(image, slider, mode, slider_alpha, label_mode, anno_mode)
#     print(image_out)
#     im = Image.fromarray(image_out)
#     im.save("your_file.jpeg")
#     # im1 = image_out.save("geeks.jpg")