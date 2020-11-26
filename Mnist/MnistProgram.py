import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

def create_model():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  return model

model = create_model()

checkpoint_path = "FredeTraining/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print("Restored model, accuracy: {:5.3f}%".format(100 * test_acc))


# predictions = model.predict(test_images)
# # Modellens gæt på første stykke tøj.
# print("Modellens Gæt:", np.argmax(predictions[0]))
# # Hvad første stykke tøj faktisk var.
# print("Korrekte stykke tøj:", test_labels[0])


COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, correct_label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.get_cmap('binary'))
  plt.title("Programmets gæt: " + guess, fontsize=18)
  plt.xlabel("Objektet er: " + correct_label, fontsize=18)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Vælg et tal: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Prøv igen...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)