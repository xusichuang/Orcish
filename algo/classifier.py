import os
current_directory = os.getcwd()

# 打印当前工作目录路径
print("当前工作目录:", current_directory)

from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch import nn
from PIL import Image
import numpy as np


processor = ViTImageProcessor.from_pretrained("./model/checkpoint-1800")
model = ViTForImageClassification.from_pretrained("./model/checkpoint-1800")
# processor = ViTImageProcessor.from_pretrained("../model/checkpoint-2600")
# model = ViTForImageClassification.from_pretrained("../model/checkpoint-2600")
def get_label(
        image : np.array
        ):
    
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    # print('***********************************')
    # print(logits)
    # print(logits.shape)

    res = nn.Softmax(-1)(logits)
    max_value, max_index = torch.max(res,dim=-1)
    # print('argmax:')
    print(max_value)
    if max_value.item()<0.4:
        return "others"
    # print(res)
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def write_label_into_picture(input_path):
    import cv2
    pic_names = os.listdir(input_path)
    for pic_name in pic_names:
        img_path = os.path.join(input_path,pic_name)
        print(img_path)
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = Image.open(img_path)
        label = get_label(img)
        print(label)
        Add_pic = np.array(img.copy())
        # output_img222 = Image.fromarray(Add_pic)
        # output_img222.save(f'/root/orcish/pic_output/test_pic_output1/123123{pic_name}')

        cv2.putText(Add_pic,label,(50,200),cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), 5)
        
        output_img = Image.fromarray(Add_pic)
        if not os.path.exists('/root/orcish/pic_output/test_pic_output2'):
            os.makedirs('/root/orcish/pic_output/test_pic_output2')
        output_img.save(f'/root/orcish/pic_output/test_pic_output2/{pic_name}')
        # break

# print(__name__)
if __name__ == "__main__":
    # image = Image.open('../orcish/cat.jpg')
    # predicted_label = get_label(image)
    # print(predicted_label)

    write_label_into_picture('/root/orcish/pic_input/test_pic2')