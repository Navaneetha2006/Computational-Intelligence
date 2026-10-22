from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('disaster_modeltrain.keras')

# ✅ Correct class order
classes = ['destroyed','major_damage','minor_damage','no_damage']

img_path = input("Enter image path: ").strip().replace('"','')

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

result = classes[np.argmax(prediction)]

print("Prediction:", result)