
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter

from PIL import Image

# thư viện os giúp load nhiều bức ảnh trong 1 forder
forderPath = "mjsynth_sample"

# kiểm tra toàn bộ tên file trong forderpath
dataset = os.listdir(forderPath)

image_paths = []
image_texts = []

data_folder = "mjsynth_sample"

for path in os.listdir(data_folder):
    image_paths.append(data_folder + "/" + path)
    image_texts.append(path.split("_")[1])

# print(image_paths[:10])
# print(image_texts[:10])

corrupt_images = []

# chuyển tất cả ảnh sang ảnh gray
for path in image_paths:
    try:
        pass
    except:
        corrupt_images.append(path)
# print(corrupt_images)
# print(len(corrupt_images))

# loại bỏ các đường dẫn ảnh bị hỏng khỏi danh sách image_paths và image_texts
for path in corrupt_images:
    corrupt_index = image_paths.index(path)
    del image_paths[corrupt_index]
    del image_texts[corrupt_index]

# Sử lý đầu ra của ảnh
# map(str, image_texts): Chuyển đổi tất cả các phần tử trong danh sách image_texts thành kiểu chuỗi (string)
# "".join(map(str, image_texts)): Kết nối tất cả chuỗi trong danh sách đã chuyển đổi thành một chuỗi duy nhất
# set: loại bỏ các ký tự trùng lặp
vocab = set("".join(map(str, image_texts)))
# print(sorted(vocab))

#tìm và lưu độ dài của chuỗi dài nhất trong danh sách image_texts vào biến max_label_len
max_label_len = max([len(str(text)) for text in image_texts])
# print(max_label_len)

char_list = sorted(vocab)
# Hàm encode_to_labels chuyển đổi mỗi chuỗi ký tự trong image_texts thành một chuỗi chỉ số (index)
# tương ứng với char_list và đệm nó thành chuỗi có độ dài cố định max_label_len (độ dài của chuỗi dài nhất trong image_texts)
def encode_to_labels(txt):
    # mã hóa từng từ đầu ra thành các chữ số
    dig_lst = []
    # Duyệt qua từng ký tự char cùng với chỉ số index của chúng trong chuỗi txt
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst

# chia danh sách image_paths và padded_image_texts thành hai phần: tập huấn luyện (train) và tập kiểm định (validation).
train_image_paths = image_paths[: int(len(image_paths) * 0.90)]
train_image_texts = image_texts[: int(len(image_texts) * 0.90)]

val_image_paths = image_paths[int(len(image_paths) * 0.90):]
val_image_texts = image_texts[int(len(image_texts) * 0.90):]

# trình tạo dữ liệu tùy chỉnh để tải và xử lý trước dữ liệu hình ảnh và văn bản theo lô một cách hiệu quả
class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    # Trả về tổng số lô mà trình tạo sẽ cung cấp, được tính bằng cách chia tổng số hình ảnh cho kích thước lô.
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    # Trả về một loạt hình ảnh và nhãn tương ứng của chúng dựa trên chỉ mục idx đã cho
    def __getitem__(self, idx):
        # Lặp qua từng đường dẫn hình ảnh và nhãn tương ứng trong lô hiện tại
        batch_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_texts = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        training_txt = []
        train_label_length = []
        train_input_length = []

        for im_path, text in zip(batch_paths, batch_texts):

            try:
                # chuyển đổi đối tượng text thành một chuỗi (nếu chưa phải),
                # và strip() loại bỏ các khoảng trắng không mong muốn trước và sau chuỗi đó
                text = str(text).strip()
                # giảm kích thước dữ liệu, tập trung vào thông tin cấu trúc hơn và đồng nhất dữ liệu
                img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)

                #  actually returns h, w
                h, w = img.shape

                # if height less than 32
                if h < 32:
                    # tạo một mảng chứa các giá trị 255 (trắng) có kích thước (32-h, w)
                    # tức thêm một phần "trắng" vào phía dưới ảnh nhằm đạt chiều cao 32
                    add_zeros = np.ones((32 - h, w)) * 255
                    # Hàm np.concatenate() mặc định nối các mảng theo hàng (theo trục 0)
                    img = np.concatenate((img, add_zeros))
                    h = 32

                # if width less than 128
                if w < 128:
                    # np.ones(m, n)  tạo mảng có kích thước m hàng n cột và tất cả giá trị bằng 1s
                    add_zeros = np.ones((h, 128 - w)) * 255
                    # axis=1 để nối các mảng theo cột
                    img = np.concatenate((img, add_zeros), axis=1)
                    w = 128

                # if width is greater than 128 or height greater than 32
                if w > 128 or h > 32:
                    img = cv2.resize(img, (128, 32))

                # thêm một trục mới vào chỉ số 2 của mảng img
                img = np.expand_dims(img, axis=2)

                # chuẩn hóa dữ liệu giúp giảm bớt các biến thiên lớn về tỷ lệ giá trị điểm ảnh,
                # đồng thời giữ lại thông tin quan trọng trong ảnh
                img = img / 255.

                images.append(img)
                training_txt.append(encode_to_labels(text))
                train_label_length.append(len(text))
                train_input_length.append(31)
            except:

                pass

        return [np.array(images),
                # hàm pad_sequences giúp đảm bảo rằng tất cả các chuỗi trong danh sách training_txt có cùng độ dài
                # training_txt: Là danh sách các chuỗi cần đồng bộ độ dài.
                # maxlen: Độ dài tối đa của mỗi chuỗi sau khi đã được đệm.
                # padding: Tham số này chỉ định phần đệm sẽ được thêm vào đầu hoặc cuối chuỗi. Giá trị 'post' ngụ ý việc đệm sẽ được thực hiện ở cuối chuỗi.
                # value: Giá trị sẽ được sử dụng để đệm. Trong trường hợp này là len(char_list), tức độ dài của danh sách ký tự.
                pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list)),
                # Chuyển danh sách độ dài đầu vào huấn luyện (train_input_length) thành một mảng NumPy.
                np.array(train_input_length),
                # Tạo một mảng các số không có cùng độ dài với số lượng hình ảnh.
                # Nó được sử dụng làm trình giữ chỗ cho các nhãn hoặc giá trị đích trong quá trình đào tạo
                # train_label_length Chuyển danh sách độ dài nhãn huấn luyện (train_label_length) thành một mảng NumPy.
                np.array(train_label_length)], np.zeros(len(images))

batch_size = 256
train_generator = My_Generator(train_image_paths, train_image_texts, batch_size)
val_generator = My_Generator(val_image_paths, val_image_texts, batch_size)
# print(train_generator[0])
# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_4)
# tăng tốc quá trình huấn luyện mô hình học sâu
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
# 4, 32, 64
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
# 2, 32, 64
conv_7 = Conv2D(64, (2, 2), activation='relu')(pool_6)
# 1, 31, 64
# Lớp Lambda trong Keras có tác dụng gói các biểu thức tùy ý thành một đối tượng Layer
# Hàm K.squeeze() loại bỏ các chiều có kích thước 1 từ tensor,
# đơn giản hóa kích thước của tensor conv_7. Kết quả của lớp Lambda được lưu vào biến squeezed
squeezed = Lambda(lambda x: tf.squeeze(x, 1))(conv_7)
# 31, 64

# Đầu ra từ lớp RNN sẽ bao gồm các giá trị xác suất cho mỗi nhãn tương ứng với mỗi đặc điểm đầu vào (input feature)
# Đầu ra của lớp blstm_2 sẽ có kích thước (batch_size, timesteps, features)
# batch_size là kích thước batch (số mẫu trong một batch).
# timesteps là số bước thời gian trong chuỗi đầu vào.
# features là số đặc trưng đầu ra của lớp LSTM hai chiều (trong trường hợp này, số đặc trưng là 128 LSTM units x 2 hướng = 256 đặc trưng).
blstm_1 = Bidirectional(keras.layers.LSTM(128, return_sequences=True))(squeezed)
blstm_2 = Bidirectional(keras.layers.LSTM(128, return_sequences=True))(blstm_1)

outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

# mô hình được sử dụng tại thời điểm thử nghiệm
act_model = Model(inputs, outputs)
act_model.summary()
# CTC loss yêu cầu 4 tham số để tính toán đó là:
# đầu ra dự đoán, nhãn thật, độ dài chuỗi đầu vào cho LSTM và độ dài nhãn thật
# Sau khi chúng ta custom được hàm loss thì sẽ pass nó vào trong mô hình trên
# Tạo một đầu vào labels với chiều dài tối đa là max_label_len. Đây là nơi lưu trữ nhãn thực tế cho mỗi mẫu dữ liệu
labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
# Tạo một đầu vào input_length với kiểu dữ liệu là số nguyên 64-bit. Độ dài đầu vào của mạng nơ-ron sẽ được lưu trữ ở đây.
input_length = Input(name='input_length', shape=[1], dtype='int64')
# Tạo một đầu ra label_length với kiểu dữ liệu là số nguyên 64-bit. Độ dài của nhãn thực tế sẽ được lưu trữ ở đây.
label_length = Input(name='label_length', shape=[1], dtype='int64')

# nhận đối số `args` bao gồm `y_pred` (dự đoán của mô hình),
# `labels` (nhãn thực tế), `input_length` (độ dài đầu vào của mạng nơ-ron)
# `label_length` (độ dài của nhãn thực tế).
# Hàm này sử dụng `K.ctc_batch_cost` để tính toán chi phí CTC trên toàn bộ batch.
# Các giá trị này sẽ được sử dụng trong quá trình huấn luyện mô hình để tối ưu hóa trọng số và giảm độ lỗi.
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # tính toán chi phí ctc
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_loss(y_true, y_pred):
    return y_pred

loss_out = Lambda(ctc_lambda_func,
                  output_shape=(1,),
                  name='ctc')([outputs, labels, input_length, label_length])

# Model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

file_path = "C_LSTM_best.hdf5"

# Use the named function when compiling the model
model.compile(loss={'ctc': ctc_loss}, optimizer='adam')
#  lưu trữ mô hình
checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                             # validation (val_loss) để theo dõi và xác định mô hình tốt nhất
                             monitor='val_loss',
                             verbose=1,
                             # Chỉ lưu mô hình có hiệu năng tốt nhất (dựa trên val_loss)
                             save_best_only=True,
                             # giảm thiểu val_loss để tìm mô hình tốt nhất
                             mode='min')

callbacks_list = [checkpoint]

epochs = 23

history = model.fit(train_generator,
                    epochs = epochs,
                    # Số bước trong mỗi epoch. Trong trường hợp này, nó bằng với tổng số ảnh huấn luyện chia cho kích thước của mỗi lô (batch_size).
                    steps_per_epoch = len(train_image_paths) // batch_size,
                    # Trình tạo dữ liệu kiểm định, sử dụng để kiểm tra sự tiến bộ và hiệu suất của mô hình sau mỗi epoch.
                    validation_data=val_generator,
                    # Số bước trong quá trình kiểm định.
                    validation_steps = len(val_image_paths) // batch_size,
                    verbose = 1,
                    # danh sách các callback được sử dụng trong quá trình huấn luyện.
                    # Trong trường hợp này, chúng ta đã thêm vào ModelCheckpoint để lưu mô hình có hiệu năng tốt nhất dựa trên val_loss
                    callbacks = callbacks_list,
                    # Có thể xáo trộn mỗi lô dữ liệu trong mỗi epoch hay không.
                    # True nghĩa là dữ liệu sẽ được xáo trộn, giúp cải thiện việc huấn luyện của mô hình.
                    shuffle=True)

# `model` được sử dụng để huấn luyện và dùng để tính toán hàm mất mát
# `act_model` được sử dụng để dự đoán kết quả của mô hình trên các dữ liệu đầu vào
act_model.save('model_api_functional.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

