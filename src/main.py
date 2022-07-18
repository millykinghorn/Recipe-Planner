# Milly Kinghorn
# July 2022

from tokenize import PlainToken
import keras_ocr
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

#read image from the an image path (a jpg/png file or an image url)
img = keras_ocr.tools.read('../baked_aubergine.jpg')
# Prediction_groups is a list of (word, box) tuples
prediction_groups = pipeline.recognize([img])
# Print words found
predicted_image_1 = prediction_groups[0]
for text, box in predicted_image_1:
  print(text)

#print image with annotation and boxes
output_img = keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])
plt.show()