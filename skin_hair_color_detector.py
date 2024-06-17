import stone
from json import dumps
import numpy as np
from PIL import ImageColor
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
from timeit import default_timer

class HairSegmentModel(nn.Module):
    def __init__(self):
        super(HairSegmentModel,self).__init__()
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights=0, progress=1, num_classes=2)
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

def personal_color(skin_rgb, hair_rgb):
    skin_r, skin_g, skin_b = skin_rgb
    hair_r, hair_g, hair_b = hair_rgb
    color = "Cool Summer"

    if (236.634 < skin_r < 251.07) and (190.82 < skin_g <213.898) and (156.34 < skin_b < 204.588) and (82.068 < hair_r <135.233) and (55.222 < hair_g < 112.443) and (46.35 < hair_b < 95.424):
        color = "Warm Spring"
    if (240.036 < skin_r < 247.935) and (217.87 < skin_g <235.789) and (201.65 < skin_b <229.502) and (207.11 < hair_r < 220.809) and (199.462 < hair_g <209.827) and (185.857 < hair_b < 223.913):
        color = "Light Spring"
    if (222.57 < skin_r < 232.57) and (169.053 < skin_g < 179.053) and (138.618 < skin_b < 148.618) and ( 30.91< hair_r < 40.91) and (26.95 < hair_g < 36.95) and (33.027 < hair_b < 43.027):
        color = "Clear Spring"
    if (225.96 < skin_r < 236.96) and (179.66 < skin_g < 189.66) and (160.63 < skin_b < 170.63) and (82.17 < hair_r < 233.83) and (83.87 < hair_g < 112.443) and (68.539 < hair_b < 95.424):
        color = "Light Summer"
    if (221.53 < skin_r < 231.53) and (201.212 < skin_g < 211.212) and (158.026 < skin_b < 168.026) and (56.35 < hair_r < 129.498) and (38.509 < hair_g < 118.76) and (32.609 < hair_b < 112.08):
        color = "Soft Summer"
    if  (171.69 < skin_r < 235.55) and (128.115 < skin_g < 177.84) and (106.915 < skin_b < 145.87) and (12.667 < hair_r < 22.667) and (13.89 < hair_g < 23.89) and (14.287 < skin_b < 24.287):
        color = "Cool Summer" 
    if (217.086 < skin_r < 227.086) and (172.84 < skin_g < 182.84) and (140.87 < skin_b < 150.87) and (83.62 < hair_r <164.22) and (62.134 < hair_g <119.28) and (51.35 < hair_b < 93.144):
        color = "Soft Autumn"
    if (198.96 < skin_r < 208.96) and (142.296 < skin_g < 152.296) and (107.128 < skin_b < 119.128) and (17.667 < hair_r < 71.126) and (18.89 < hair_g < 51.307) and (19.287 < hair_b < 33.89):
        color = "Deep Autumn"
    if (173.368 < skin_r < 186.368) and (123.39 < skin_g < 133.39) and (95.1 < skin_b < 105.1) and (123.48 < hair_r < 162.109) and (62.97 < hair_g < 120.258) and (33.37 < hair_b < 113.376):
        color = "Warm Autumn"
    if (221.107 < skin_r < 231.107) and (166.97 < skin_g < 176.97) and (139.054 < skin_b < 149.54) and (12.667 < hair_r < 22.667) and (13.89 < hair_g <23.89) and (14.287 < hair_b <24.287):
        color = "Deep Winter"
    if (240.454 < skin_r < 250.454) and (202.065 < skin_g < 212.065) and (186.89 < skin_b < 196.89) and (70.56 < hair_r < 150.524) and (70.56 < hair_g < 109.96) and (72.56 < hair_b < 82.66):
        color = "Clear Winter"
    if (217.42 < skin_r < 227.42) and (141.424 < skin_g < 151.424) and (162.488 < skin_b < 172.488) and (49 < hair_r < 59) and (32.069 < hair_g < 42.069) and (29.116 < hair_b < 39.116):
        color = "Cool Winter"
    if (180 < skin_r < 200) and (141.424 < skin_g < 151.424) and (120 < skin_b < 160) and (0 < hair_r < 20) and (0 < hair_g < 20) and (0 < hair_b < 20.116):
        color = "Light Summer"
    if (180 < skin_r < 200) and (141.424 < skin_g < 155.424) and (120 < skin_b < 160) and (140 < hair_r < 180) and (140 < hair_g < 160) and (130 < hair_b < 150):
        color = "Light Spring"
    if (160 < skin_r < 180) and (145.424 < skin_g < 161.424) and (120 < skin_b < 160) and (20 < hair_r < 30) and (15 < hair_g < 25) and (10 < hair_b < 25.116):
        color = "Clear Winter"
    if (180 < skin_r < 200) and (131.424 < skin_g < 141.424) and (110 < skin_b < 140) and (0 < hair_r < 20) and (0 < hair_g < 20) and (0 < hair_b < 10.116):
        color = "Deep Autumn"
    return color

def detect_face(image_path):
    # Load the pre-trained Haar cascade file for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return True
    else:
        return False

def get_skin_color(image_path):
    result = stone.process(image_path, image_type='auto', n_dominant_colors=1, return_report_image=True)
    report_images = result.pop("report_images") 
    face_list = [id for id in report_images.keys()]
    face_id = face_list[0]
    # Uncomment the line below to show image with measurements
    # stone.show(report_images[face_id])  

    result_json = dumps(result)
    results = result_json.split(',')
    skin_tone = "#000000"
    for item in results:
        if "dominant" in item:
            skin_tone = item.split(':')[-1].replace('\"','').strip()

    hex_code = skin_tone.lstrip('#')
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return r, g, b

def get_hair_mask(image_path, checkpoint_path='D:\modelvytec\hair_detect.pt'):
    if isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)
    else:
        img = Image.open(image_path)
    
    preprocess = transforms.Compose([transforms.Resize((512, 512), 2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    Xtest = preprocess(img)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = HairSegmentModel()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        device = torch.device('cpu') # cpu | cuda
        model.to(device)
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        ytest = ypos >= yneg
    
    mask = ytest.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_hair_color(image_path):
    image = cv2.imread(image_path)  
    if image is None:
        print("Failed to load the image from:", image_path)
        return
    if image.shape[0] < 1 or image.shape[1] < 1:
        print("Invalid image dimensions:", image.shape)
        return
    image = cv2.resize(image, (512, 512))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = get_hair_mask(rgb)
    bool_mask = mask.astype(bool)
    region = image[bool_mask]

    hair_color = []
    for c in range(3):
        unique, counts = np.unique(region[:,c], return_counts=True)
        hair_color.append(unique[counts.argmax()])

    hair_color_rgb = hair_color[::-1]
    return tuple(hair_color_rgb)
    
if __name__ == '__main__':
    image_path = 'D:\modelvytec\skincolor.jpg'  # get image path to detect
    # start = default_timer()
    skin_color, hair_color = get_skin_color(image_path), get_hair_color(image_path)
    # end = default_timer()
    print(f'Skin Tone: {skin_color} | Hair Color RGB: {hair_color}')
    # print(f"Predict time: {end-start:.2f} seconds")
