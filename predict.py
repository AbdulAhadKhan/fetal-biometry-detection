import numpy as np

def predict_class(model, data, label_dict):
    prediction = model.predict(data)
    prediction = np.where(prediction == np.amax(prediction))

    key = prediction[1][0]
    label = label_dict.get(key)

    return label

def predict_confidence(model, data, label_dict):
    prediction = model.predict(data)[0]
    prediction = np.round(prediction * 100, 3)

    class_pred = list(zip(label_dict.values(), prediction))

    return class_pred