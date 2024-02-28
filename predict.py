from keras.models import load_model
import cv2
import numpy as np

model = load_model("model_weights.h5")
img = cv2.imread(r"C:\Users\Vidhu\Downloads\WhatsApp Image 2024-02-28 at 15.21.41_575eaf34.jpg")

img = cv2.resize(img, (150, 150))
img_array = np.array(img)/255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Predicted: Dog (Probability:", prediction[0][0], ")")
else:
    print("Predicted: Cat (Probability:", 1 - prediction[0][0], ")")