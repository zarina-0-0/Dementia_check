import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_NAME = 'model.pkl'
DEMENTIA_T = {'0' : 'healthy', '1' : 'alzheimer', '2' : 'frontotemporal dementia'}


UPLOAD_FOLDER = '/home/alena/commonwin/s'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(src_img, size=(208,176), bg_color="black"): 
    src_img.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("L", size, bg_color)
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))
    return new_image


@app.route('/', methods=['GET', 'POST'])
def upload_file(result = None):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = Image.open(filename)
            new_image = np.asarray(resize_image(img))
            im_vector = np.empty(0)
            im_vector = np.concatenate([ar for ar in new_image])
            loaded_model = pickle.load(open(MODEL_NAME, 'rb'))
            result = DEMENTIA_T[str(loaded_model.predict([im_vector])[0])]
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload.html', result=result)
    return render_template('upload.html', result=result)
if __name__ == '__main__':
   app.run(debug = True)
