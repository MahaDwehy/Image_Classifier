#import necessary libraries
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.python.ops.gen_nn_ops import top_k

#using argument parser
arg_parser = argparse.ArgumentParser(description='Prediction of flower class using command line')
arg_parser.add_argument('--image_path')
arg_parser.add_argument('--model_path')
arg_parser.add_argument('--category_names')
arg_parser.add_argument('--top_k', type = int, default =3)
args = arg_parser.parse_args()

#function to process the image
def process_image(image):
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, (224, 224))
    image/=255
    return image.numpy()

# predict function
def predict(image_path, model, top_k):

    #reading and preprocessing the image
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img, axis=0)

    #make prediction
    predictions = model.predict(img)

    #take top prediction probabilities
    top_probs, top_index = tf.math.top_k(predictions, top_k)

    #corresponding probabilities and classes
    probs = top_probs.numpy()[0]
    classes = top_index.numpy()[0]

    return probs , classes 

#load the model
model = tf.keras.models.load_model(args.model_path,custom_objects={'KerasLayer':hub.KerasLayer})

#calling the predict function
probs, classes  = predict(args.image_path, model, args.top_k)

#printing results
print("Top {} Probabilities are: {}".format(args.top_k, probs))
print("Corresponding Top {} Labels are: {}".format(args.top_k, classes))

#if json to map is provided, index will be converted to class and corresponding top class will be predicted.
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    classes = [class_names[str(value+1)] for value in classes]
    print('Corresponding Top {} class names are: {} '.format(args.top_k, classes))
