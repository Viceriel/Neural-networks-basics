import os
from tensorflow.keras.models import model_from_json

def save_model(model, name):
    dir = "model"
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_json = model.to_json()
    json_file = open(dir + "/" + name + ".json", "w")
    json_file.write(model_json)
    model.save_weights(dir + "/" + name + ".h5")

def load_model(path, name):
    json_file = open(path+"/"+name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path+"/"+name+".h5")
    model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
    return model