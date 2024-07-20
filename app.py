from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('model_inception.h5')

# Define image dimensions
img_height = 224
img_width = 224

# Define class labels
class_labels = ['benign', 'malignant']  # Modify with your actual class labels

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format
    img_array /= 255.  # Rescale to [0,1]

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return class_labels[predicted_class], confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            img_path = "temp.jpg"  # Save the uploaded image temporarily
            img_file.save(img_path)
            predicted_class, confidence = predict_image(img_path)
            return jsonify({'class': predicted_class, 'confidence': float(confidence)})
        return render_template('prediction.html')
    return render_template('prediction.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        password= request.form["password"]
        if email == "admin@gmail.com" and password == "admin":
            msg = "Logged in sucsessfully"
            return render_template("prediction.html", msg=msg)
        else:
            msg = "Invalid credentials"
            return render_template("login.html", msg=msg)
    return render_template("login.html")

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/logout")
def logout():
    return render_template("home.html")

if __name__ == '__main__':
    app.run()


