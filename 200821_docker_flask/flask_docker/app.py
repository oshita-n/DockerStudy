from flask import Flask, redirect, request, render_template, url_for, send_from_directory
import tensorflow as tf
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
model = tf.keras.models.load_model('my_model.h5')

@app.route("/", methods=["GET", "POST"])
def route_access():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return str(np.argmax(model.predict(np.reshape(np.array(Image.open(os.path.join('uploads', file.filename))), (1, 28, 28)))))
    return render_template("a.html")
