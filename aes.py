import os
import requests
from pathlib import Path
import numpy as np
import clip
import torch
import random
import math
from torch import nn
from PIL import Image
from mlp import MLP

class Aesthetic_predictor():
    def __init__(self):
        self.model = None
        self.clip_model = None
        self.preprocess = None
    
    def OnInit(self):
        self.model = MLP(768)
        self.model.load_state_dict(self.load_aesthetic_model_weights())
        self.model.cuda()
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device="cuda")
    
    def preprocess_image(self, img):
        image = self.preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            return image_features

    def load_aesthetic_model_weights(self, cache="."):
        weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
        loadpath = os.path.join(cache, weights_fname)

        if not os.path.exists(loadpath):
            url = (
                "https://github.com/christophschuhmann/"
                f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
            )
            r = requests.get(url)
            
            with open(loadpath, "wb") as f:
                f.write(r.content)

        weights = torch.load(loadpath, map_location=torch.device("cpu"))
        return weights

    def aesthetic_model_normalize(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a/np.expand_dims(l2, axis)

    def get_score(self,img):
        image_features = self.preprocess_image(img)
        im_emb_arr = self.aesthetic_model_normalize(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).float().cuda())
        return prediction.item()

if __name__ == "__main__":
    
    img = Image.open('/img_path')

    aesthetic_model = Aesthetic_predictor()
    aesthetic_model.OnInit()

    image_features = aesthetic_model.preprocess_image(img)

    im_emb_arr = aesthetic_model.aesthetic_model_normalize(image_features.cpu().detach().numpy())
    prediction = aesthetic_model.model(torch.from_numpy(im_emb_arr).float().cuda())

    print(f'Aesthetic score : {prediction}')