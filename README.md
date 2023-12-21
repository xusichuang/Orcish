# orcish

# 背景：圣诞活动，用户上传一张自己的宠物图片，生成对应的兽人形象

## 效果展示
<div style="display: flex; justify-content: space-between;">
    <img src="test-scripts\output1.png" alt="Image 1" style="width: 48%;">
    <img src="test-scripts\output2.png" alt="Image 2" style="width: 48%;">
</div>
<div style="display: flex; justify-content: space-between;">
    <img src="test-scripts\output3.png" alt="Image 1" style="width: 48%;">
    <img src="test-scripts\output4.png" alt="Image 2" style="width: 48%;">
</div>

### 最终效果可见[这里](https://github.com/xusichuang/orcish/tree/main/test-scripts/output)

# 流程图

![流程图](pipeline_pic.png "Pipeline picture")

## 细节  
    1.底膜：realcartoon3d_v10.safetensors 
    
    2.lora:<lora:add_dataset2_realcartoon3d:0.7> 
    
    
    3.参数
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
    
    4.prompt
        positive_prompt
            tag + <lora:add_dataset2_realcartoon3d:0.7> + An enchanting image featuring an adorable kitten mage wearing intricate ancient robes, \
             |                                              holding an ancient staff, hard at work in her fantastical workshop, intricate runic symbols swirling around her, \
             |                                              it\'s clear that she\'s busy casting a powerful spell. Her fluffy tail sways gently as she concentrates on the task at hand, \
             v                                              adding to the whimsical atmosphere of this magical scene. '
            标签映射
                                                            

        negative_prompt
            '(nsfw:1.3),(Nude:1.3),(Naked:1.3),extra fingers,fewer fingers, (bad-hands-5:1.4),deformed,text, bad hand, extra hands, extra fingers, too many fingers, fused fingers,\
            worst reslution,low quality,(normal quality:2),lowres,signature,watermark,paintings, sketches,skin spots, skin blemishes, age spot,'

    5.标签映射
        tag格式: (a qwert animals:1.1)
        
        (1)狗、鼠不做标签映射直接用
        
        
        (2)猫、兔子做标签映射
            

            
    6.pipeline
        图片--->分类器--->label--->标签映射--->tag--->4大类动物使用txt2img、others使用img2img

        
        txt2img:使用5.的prompt

        img2img使用CLIP反推作为positive_prompt，negative_prompt使用5.
        
# 有待提升部分

- others类别风格有差别
- 加入LCM提升推理速度，LCM参数可见[这里](https://github.com/xusichuang/orcish/tree/main/test-scripts/output)