from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("./models/efficient_det.keras")

img_path = './test_images/house.png'
img = image.load_img(img_path, target_size=(320, 320))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
classes = ['car', "cat", "chair", "dog", "door"]
print(f"Predicted class: {classes[np.argmax(predictions)]}, Confidence: {np.max(predictions) * 100:.2f}%")
