from PIL import Image
from io import BytesIO
import base64
# from fastapi.exceptions import HTTPException
import io
import cv2
import requests
from typing import Tuple

def decode_base64_to_image(encoding):
    # 如果数据以data:image/开头   则需要去除 img_base64_str 中的数据 URI scheme 部分，保留实际的图像编码数据
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]

    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        # print('status_code=500    Invalid encoded image')
        raise ValueError('status_code=500    Invalid encoded image')
        # raise HTTPException(status_code=500, detail="Invalid encoded image") from e




def encode_pil_to_base64(image_path):
    img_pil = Image.open(image_path)
    # 创建一个字节流对象，用于存储图像字节流数据
    output_buffer = BytesIO()
    img_pil.save(output_buffer, format='PNG')
    # 获取图像的字节流数据
    img_byte_data = output_buffer.getvalue()
    # 将字节流数据进行base64编码  得到编码字符串
    img_base64_str = base64.b64encode(img_byte_data)
    # with open('/root/orcish/ba.txt','wb') as file:
    #     file.write(img_base64_str)
    
    return img_base64_str

###############################################################################
# img_base64_str = encode_pil_to_base64('/root/openpose_test/1.png')
# print(img_base64_str)
# img = decode_base64_to_image(img_base64_str.decode())
# print(img)
###############################################################################

###############################################################################
# # 对编码字符串进行解码
# data = img_base64_str.decode()
# data = "data:image/" + data
# print(data)
# img2 = decode_base64_to_image(data)
# print(img2)
###############################################################################


# Read Image in RGB order
'''
img_path = '/root/openpose_test/1.png'
img = cv2.imread(img_path)

# Encode into PNG and send to ControlNet
retval, bytes = cv2.imencode('.png', img)
encoded_image = base64.b64encode(bytes).decode('utf-8')
'''

class Txt2ImgRequest():
    def __init__(self,
                ip : int,
                positive_prompt : str,
                negative_prompt : str,
                size:Tuple,
                batch_size : int,
                encoded_image = None):
        '''
        ip : port
        '''
        self.width,self.height = size
        self.url = "http://127.0.0.1:%s/sdapi/v1/txt2img"%ip
        self.body = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            # "sampler_index": "Euler a",
            # "sampler_name": "Euler a",
            "sampler_index": "DPM++ 2M Karras",
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": batch_size,
            "steps": 20,
            "width": self.width,
            "height": self.height,
            "cfg_scale": 7,
            'seed':-1,
            'n_iter':1
            # "alwayson_scripts": {
            #     "controlnet": {
            #         "args": [
            #             {
            #                 "input_image": encoded_image,
            #                 "module": "canny",
            #                 "model": "control_v11p_sd15_canny [d14c016b]",
            #             }
            #         ]
            #     }
            # }
        }
    
    def sendRequest(self):
        response = requests.post(self.url, json=self.body)
        return response.json()


class Img2ImgRequest():
    def __init__(self,
                ip : int,
                positive_prompt : str,
                negative_prompt : str,
                size:Tuple,
                batch_size : int,
                encoded_image = None):
        '''
        ip : port
        '''
        self.width,self.height = size
        self.url = "http://127.0.0.1:%s/sdapi/v1/img2img"%ip
        self.body = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "init_images": [encoded_image],
            "resize_mode": 0,
            # "sampler_index": "Euler a",
            # "sampler_name": "Euler a",
            "sampler_index": "DPM++ 2M Karras",
            "sampler_name": "DPM++ 2M Karras",
            "steps": 20,
            "width": self.width,
            "height": self.height,
            'n_iter':1,
            "batch_size": batch_size,
            "cfg_scale": 7,
            "denoising_strength": 0.8,
            'seed':-1,

            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": encoded_image,
                            "module": "canny",
                            "model": "control_v11p_sd15_canny [d14c016b]",
                        }
                    ]
                }
            }
        }
    
    def sendRequest(self):
        response = requests.post(self.url, json=self.body)
        # print(response)
        return response.json()
        
class Interrogate_CLIP_Request():
    def __init__(self,
                 ip,
                 encoded_image):
        self.url = "http://127.0.0.1:%s/sdapi/v1/interrogate"%ip
        self.body = {
            "image": encoded_image,
            "model": "clip"
        }
    
    def sendRequest(self):
        # assert requests.post(self.url,json=self.body).status_code==200
        response =  requests.post(self.url,json=self.body)
        # print(response)
        return response


        