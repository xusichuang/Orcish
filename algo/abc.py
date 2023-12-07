import os

from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch import nn
from PIL import Image
import numpy as np

# 训练好的分类模型的绝对路径/root/orcish/model/checkpoint-1800
processor = ViTImageProcessor.from_pretrained("./model/checkpoint-1800")
model = ViTForImageClassification.from_pretrained("./model/checkpoint-1800")

def get_label(
        image : np.array
        ):
    
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    res = nn.Softmax(-1)(logits)
    max_value, max_index = torch.max(res,dim=-1)

    if max_value.item()<0.4:
        return "others"

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]
animals_amp = {'Ginger':'Garfield', 'Ginger_white':'orange_white_Cat', 'Siamese':'Siamese', 'British_blue':'gray_cat', 
               'Sliver_British_Shorthair':'gray_cat', 'Golden_British_Shorthair':'Garfield', 'Cow_Cat':'heibai_Cat',
               'Chinese_Li_Hua':'Chinese_Li_Hua','white_Cat':'white_Cat','black_Cat':'black_Cat', 
               'British_Short_blue_white':'huibai_Cat', 'Tortoiseshell':'G_Cat', 'American_Shorthair':'heibai_Cat',
               

               'gray_Rabbit':'black_rabbit', 'huibai_Rabbit':'heibai_Rabbit','white_Rabbit':'white_Rabbit','yellow_Rabbit':'yellow_Rabbit'}

def get_tag(image_path):
    '''
    1.SD环境
    2.底膜：realcartoon3d_v10.safetensors
    3.lora:<lora:add_dataset2_realcartoon3d:0.7>
    
    4.参数
        txt2img
            sampler:"DPM++ 2M Karras"
            step:20
            cfg_scale: 7
        
        img2img
            sampler:"DPM++ 2M Karras"
            step:20
            cfg_scale: 7
            resize_mode: 0,
            denoising_strength: 0.6,
    
    5.prompt
        positive_prompt
            tag + <lora:add_dataset2_realcartoon3d:0.7> + An enchanting image featuring an adorable kitten mage wearing intricate ancient robes, \
             |                                              holding an ancient staff, hard at work in her fantastical workshop, intricate runic symbols swirling around her, \
             |                                              it\'s clear that she\'s busy casting a powerful spell. Her fluffy tail sways gently as she concentrates on the task at hand, \
             v                                              adding to the whimsical atmosphere of this magical scene. '
            标签映射
                                                            

        negative_prompt
            '(nsfw:1.3),(Nude:1.3),(Naked:1.3),extra fingers,fewer fingers, (bad-hands-5:1.4),deformed,text, bad hand, extra hands, extra fingers, too many fingers, fused fingers,\
            worst reslution,low quality,(normal quality:2),lowres,signature,watermark,paintings, sketches,skin spots, skin blemishes, age spot,'

    6.标签映射
        tag格式: (a qwert animals:1.1)
        
        (1)狗、鼠不做标签映射直接用
        
        
        (2)猫、兔子做标签映射
            

            
    7.pipeline
        图片--->分类器--->label--->标签映射--->tag--->4大类动物使用txt2img、others使用img2img

        
        txt2img:使用5.的prompt

        img2img使用CLIP反推作为positive_prompt，negative_prompt使用5.
        
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

                
        clip_response = Interrogate_CLIP_Request(port,encoded_image=encoded_image.decode()).sendRequest()
        # 将clip_response中的内容（字符串格式），转换成json
        prompt_json = json.loads(clip_response.text)
        positive_prompt = ' <lora:Detached1-10_realcartoon3d_Model:0.7>' + prompt_json['caption']

    '''
    image = Image.open(image_path)
    label = get_label(image)
    if label in animals_amp:
        tag = animals_amp[label]
    else:
        tag = label
        return tag
    return f'(a qwert {tag}:1.2),'

if __name__ == "__main__":
    tag = get_tag('/root/orcish/test-scripts/output2.png')
    print(tag)