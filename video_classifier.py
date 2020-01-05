import cv2
import pickle

import tensorflow as tf
import numpy as np

from predict import predict_confidence

def create_string(predictions):
    result = ''
    
    for prediction in predictions:
        result += '{label}: {value:.2f}%\n'.format(label=prediction[0], value=prediction[1])

    return result

def prepare(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_CUBIC)
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    
    return frame

def classify_frames(model, video_path, label_dict):
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()
        pred_str = ''

        frame_data = prepare(frame)
        predictions = predict_confidence(model, frame_data, label_dict)
        pred_str = create_string(predictions)

        font = cv2.FONT_HERSHEY_PLAIN

        y0, dy = 30, 30
        for i, line in enumerate(pred_str.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame, line, (0, y ), font, 2, (86, 86, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('video', frame)
        if cv2.waitKey(26) & 0xFF is ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

model = tf.keras.models.load_model('./models/alex_net_rescaled.h5')
video = './Data/Real/Fetal CNS Live Ultrasound Scan.mp4'
labels = {}

with open('classes.pickle', 'rb') as fp:
    labels = pickle.load(fp)

# print(model.layers[0].input_shape)

classify_frames(model, video, labels)