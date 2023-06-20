
import os
from tensorflow import keras
from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np

model = keras.models.load_model('model_api_functional.h5')
forderPath = "mjsynth_sample"
dataset = os.listdir(forderPath)
image_paths = []
image_texts = []
data_folder = "mjsynth_sample"
for path in os.listdir(data_folder):
    image_paths.append(data_folder + "/" + path)
    image_texts.append(path.split("_")[1])
vocab = set("".join(map(str, image_texts)))
char_list = sorted(vocab)

app = Flask(__name__)
@app.route('/process_image', methods=['POST'])
def process_image():
    # Đọc dữ liệu ảnh từ request
    image_byte = request.get_data()

    # Chuyển đổi đối tượng byte thành ảnh
    image = cv2.imdecode(np.frombuffer(image_byte, np.uint8), -1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ### actually returns h, w
    h, w = img.shape
    ### if height less than 32
    if h < 32:
        add_zeros = np.ones((32 - h, w)) * 255
        img = np.concatenate((img, add_zeros))
        h = 32
    ## if width less than 128
    if w < 128:
        add_zeros = np.ones((h, 128 - w)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
        w = 128
    ### if width is greater than 128 or height greater than 32
    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=2)
    # Normalize each image
    img = img / 255.

    # predict outputs on validation images
    prediction = model.predict(np.array([img]))
    # use CTC decoder
    out = keras.backend.get_value(keras.backend.ctc_decode(prediction,
                                                           input_length=np.ones(prediction.shape[0]) * prediction.shape[ 1],
                                                           greedy=True)[0][0])
    text_final = []
    print(out)
    ## get the final text
    for x in out:
        print("predicted text = ", end='')
        for p in x:
            if int(p) != -1:
                text = char_list[int(p)]
                text_final.append(text)
    print(text_final)
    return jsonify(text_final)

if __name__ == "__main__":
    app.run()





