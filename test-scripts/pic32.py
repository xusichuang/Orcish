
from algo import SD_FASTAPI
import math
from PIL import Image
import base64
import io
import requests
from typing import Tuple
from tqdm import tqdm
import os
# 一次性生成32种动物，每种动物4张图片脚本

def save_and_concat_pic(response,tag):
    image_list = []
    print('生成的图片数量：',len(response['images']))
    # 将response中的图片保存到本地和列表中
    if not os.path.exists(f'./test-scripts/data/{tag}'):
        os.makedirs(f'./test-scripts/data/{tag}')
    for i,result in enumerate(response['images'],start=1):
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        image_list.append(image)
        image.save(f'./test-scripts/data/{tag}/output{i}.png')

    
    # original_img = original_img.resize(image_list[0].size)
    # image_list.append(original_img)
    image_list_len = len(image_list)

    # 获取图片的尺寸
    width, height = image_list[0].size
    # 列表中第cnt张图片
    img_cnt = 0
    # 判断图片数量并将其拼接到一张图中
    if image_list_len<=3:
        new_image = Image.new('RGB', (width * image_list_len, height))
        for i,img in enumerate(image_list):
            new_image.paste(img,(i*width,0))
    else:
        if image_list_len==5 or image_list_len==6:
            new_image = Image.new('RGB', (width * 3, height * 2))
            for j in range(0,height*2,height):
                for i in range(0,width*3,width):
                    if(img_cnt==image_list_len):
                        break
                    new_image.paste(image_list[img_cnt],(i,j))
                    img_cnt +=1

        else:
            line = col = math.ceil(math.sqrt(image_list_len))
            new_image = Image.new('RGB', (width * col, height * line))
            
            for j in range(0,height*line,height):
                for i in range(0,width*col,width):
                    if(img_cnt==image_list_len):
                        break
                    new_image.paste(image_list[img_cnt],(i,j))
                    img_cnt +=1

    # 保存拼接后的图片
    new_image.save(f'./test-scripts/output/{tag}.png')

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
            "sampler_index": "DPM++ 2M Karras",
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": 4,
            "steps": 30,
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


tags = ['G_Cat','heibai_Cat','black_Cat','Chinese_Li_Hua','orange_white_Cat',\
       'Garfield','huibai_Cat','gray_Cat','Siamese','white_Cat', \
       
        'Bichon','black_Dog','Border_Collie','Bulldog','Corgi','Golden_Retriever',\
        'Siberian_Husky','Pomeranian','Poodle','Samoyed','Shiba_Inu',\
        
        'black_Hamster','white_Hamster','yellow_hamster',\
        
        'heibai_Rabbit','black_Rabbit','white_Rabbit','yellow_Rabbit'
        ]

# tags = ['Chinese_Li_Hua_Cat']

prompt = ' <lora:add_dataset2_realcartoon3d:0.7>,An enchanting image featuring an adorable kitten mage wearing intricate ancient robes, \
holding an ancient staff, hard at work in her fantastical workshop, intricate runic symbols swirling around her, \
it\'s clear that she\'s busy casting a powerful spell. Her fluffy tail sways gently as she concentrates on the task at hand, \
adding to the whimsical atmosphere of this magical scene. '

negative_prompt = '(nsfw:1.3),(Nude:1.3),(Naked:1.3),extra fingers,fewer fingers, (bad-hands-5:1.4),deformed,text, bad hand, extra hands, extra fingers, too many fingers, fused fingers,worst reslution,low quality,(normal quality:2),lowres,signature,watermark,paintings, sketches,skin spots, skin blemishes, age spot,'

for tag in tqdm(tags, desc="Processing tags", unit="tag"):
    positive_prompt = f'(a qwert {tag}:1.1),' + prompt
    print(positive_prompt)
    response = Txt2ImgRequest(8990,positive_prompt,negative_prompt,(640,640),1).sendRequest()
    save_and_concat_pic(response,tag)