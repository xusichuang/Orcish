# orcish

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
