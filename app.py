from flask import Flask, render_template, request
from PIL import Image
from mlp import MLP
from aes import Aesthetic_predictor
import os

app = Flask(__name__)
aesthetic_model = Aesthetic_predictor()
aesthetic_model.OnInit()

@app.route("/", methods=['GET'])
def main():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        if imagefile:
            image_path = "./static/" + imagefile.filename
            if not os.path.isfile(image_path):
                imagefile.save(image_path)

            img = Image.open(image_path)
            height = img.size[0]
            width = img.size[1]
            score = str(aesthetic_model.get_score(img))

            return render_template('index.html', prediction=score, img_path=image_path) 
        else:
            return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)