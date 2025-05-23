import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model('gender_classification_model.h5')

img_path = "test.jpg"

img_size = (100, 100)

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)  # 

prediction = model.predict(img_array)

predicted_class_index = int((prediction > 0.5).astype(int)[0][0])

class_labels = {0: 'Female', 1: 'Male'} 

predicted_class = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class}")
print(f"Prediction Score: {prediction[0][0]:.4f}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()
