from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import io
import base64
from PIL import Image
import numpy as np
from neural_style_tutorial import run_style_transfer, image_loader, imshow
import torchvision.models as models
import torchvision.transforms as transforms
import torch

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@127.0.0.1:3306/intent'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.dbexit()90.9'
# db = SQLAlchemy(app)

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)

def convertImage(img):
	img_str = re.search(r'base64,(.*)',img).group(1)
	img_byte = str.encode(img_str)
	img_data = base64.decodebytes(img_byte)
	# save the image
	with open('user_draw.jpg','wb') as output:
		output.write(img_data)

	png = Image.open('user_draw.jpg')
	png.load() # required for png.split()

	background = Image.new("RGB", png.size, (255, 255, 255))
	background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

	background.save('user_draw.jpg', 'JPEG', quality=80)

	img = io.BytesIO(img_data)

	return np.array(Image.open(img))[:,:,3]/255.

@app.route('/')
@nocache
def index():
	return render_template('index.html')

@app.route('/mix/', methods=['GET', 'POST'])
@nocache
def mix():

	image_b64 = request.values['imageBase64']
	convertImage(image_b64)

	# NST

	use_cuda = torch.cuda.is_available()
	dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

	style_img = image_loader("example.jpg").type(dtype)
	content_img = image_loader("user_draw.jpg").type(dtype)
	print(style_img.shape)
	print(content_img.shape)

	cnn = models.vgg19(pretrained=True).features

	# move it to the GPU if possible:
	if use_cuda:
		cnn = cnn.cuda()

	input_img = content_img.clone()

	output = run_style_transfer(cnn, content_img, style_img, input_img)

	# imshow(output)
	# plt.savefig('static/result.jpg')

	image = output.clone().cpu()
	image = image.view(3, 128, 128)
	image = transforms.ToPILImage()(image)
	image.save('static/result.jpg', 'JPEG')


	return "check"

@app.route('/test', methods=['GET', 'POST'])
def test():
	return "yo"

if __name__ == "__main__":
    app.run(debug=True)