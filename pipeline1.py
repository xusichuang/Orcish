# import os
# current_directory = os.getcwd()

# # 打印当前工作目录路径
# print("当前工作目录:", current_directory)

from algo.classifier import *
from algo.classifier import get_label

from PIL import Image
import requests
from algo.SD_FASTAPI import encode_pil_to_base64
from algo.SD_FASTAPI import decode_base64_to_image
from algo.SD_FASTAPI import Txt2ImgRequest,Img2ImgRequest,Interrogate_CLIP_Request
import cv2
import base64
import io
import math
import json 
import glob


######################################################################
                        # SD环境
                        # 底膜 realcartoon3d_v10.safetensors
######################################################################


positive_prompt_list = [' <lora:Detached1-10_realcartoon3d_Model:0.7>,An enchanting image featuring an adorable kitten mage wearing intricate ancient robes, \
                        holding an ancient staff, hard at work in her fantastical workshop, intricate runic symbols swirling around her, \
                        it\'s clear that she\'s busy casting a powerful spell. Her fluffy tail sways gently as she concentrates on the task at hand, \
                        adding to the whimsical atmosphere of this magical scene.',""]


negative_prompt = '(nsfw:1.3),(Nude:1.3),(Naked:1.3),extra fingers,fewer fingers, \
                    (bad-hands-5:1.4),deformed,text, bad hand, extra hands, extra fingers, \
                    too many fingers, fused fingers,worst reslution,low quality,(normal quality:2),\
                    lowres,signature,watermark,paintings, sketches,skin spots, skin blemishes, age spot,'


animals_amp = {'Ginger':'Ginger', 'Ginger_white':'Ginger_white', 'Siamese':'Siamese', 'British_blue':'gray_cat', 
               'Sliver_British_Shorthair':'gray_cat', 'Golden_British_Shorthair':'Ginger', 'Cow_Cat':'heibai_Cat',
               'Chinese_Li_Hua':'Chinese_Li_Hua','white_Cat':'white_Cat','black_Cat':'black_Cat', 
               'British_Short_blue_white':'huibai_Cat', 'Tortoiseshell':'Tortoiseshell', 'American_Shorthair':'heibai_Cat',
               
               'gray_Rabbit':'black_rabbit', 'huibai_Rabbit':'heibai_Rabbit','white_Rabbit':'white_Rabbit','yellow_Rabbit':'yellow_Rabbit'}

def save_and_concat_pic(original_img,response,output_pic_dir):
    image_list = []
    print('生成的图片数量：',len(response['images']))
    # 将response中的图片保存到本地和列表中
    for i,result in enumerate(response['images'],start=1):
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
        image_list.append(image)
        if not os.path.exists(output_pic_dir):
            os.makedirs(output_pic_dir)
        image.save(f'{output_pic_dir}/output{i}.png')

    
    original_img = original_img.resize(image_list[0].size)
    image_list.append(original_img)
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
    new_image.save(f'{output_pic_dir}/concat_pic.png')


def call_SDAPI(img_path,output_pic_dir,port,size,batch_size):
    try:
        original_img = Image.open(img_path)
        label = get_label(original_img)
        print('original_label:',label)
        if label in animals_amp:
            label = animals_amp[label]
        print('map_label:',label)

        if label == 'others':

            # positive_prompt = positive_prompt_list[1]

            encoded_image = encode_pil_to_base64(img_path)
            clip_response = Interrogate_CLIP_Request(port,encoded_image=encoded_image.decode()).sendRequest()
            
            # 将clip_response中的内容（字符串格式），转换成json
            prompt_json = json.loads(clip_response.text)
            # positive_prompt = '<lora:add_dataset2_realcartoon3d:0.7>' + prompt_json['caption']
            positive_prompt = prompt_json['caption']
            print(positive_prompt)
            a = 'In a magical realm, ethereal energies dance, forming intricate runes that weave an enchanting tapestry of shimmering hues. The air resonates with silent incantations, and otherworldly sigils flicker, casting an ephemeral glow. A sublime manifestation of pure magic creates a mystical atmosphere, evoking an enchanted world beyond tangible reality.'
            # positive_prompt= a + '(a small yellow chamelon sitting on a branch of a tree in a hand with a blurred background, David Young Cameron, tiny, a jigsaw puzzle, optical illusion:1.2)'
            # positive_prompt='<lora:add_dataset2_realcartoon3d:0.7>a small yellow chamelon sitting on a branch of a tree in a hand with a blurred background, David Young Cameron, tiny, a jigsaw puzzle, optical illusion'
            # positive_prompt = clip_response[]
            # print(encoded_image)
            positive_prompt += a
            response = Img2ImgRequest(port,positive_prompt,negative_prompt,size,batch_size,encoded_image=encoded_image.decode()).sendRequest()
        else:
            positive_prompt = f'(a qwert {label}:1.1),' + positive_prompt_list[0]
            # print(positive_prompt)
            response = Txt2ImgRequest(port,positive_prompt,negative_prompt,size,batch_size).sendRequest()

        # result = response['images'][0]
        # # print(result)
        # image = Image.open(io.BytesIO(base64.b64decode(result)))
        
        save_and_concat_pic(original_img,response,output_pic_dir)
        # concat_pic(original_img)
    except Exception as e:
        raise ValueError(e)



if __name__ == "__main__":

    call_SDAPI(img_path='./222.jpg',output_pic_dir='./pic_output',port=8990,size=(640,640),batch_size=4)

