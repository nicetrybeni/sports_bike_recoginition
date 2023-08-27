from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from PIL import Image
import matplotlib.pyplot as plt

# Set the path to the training images for the specific category (r15)
train_folder_path = "C:/Users/BENI/Desktop/sports_bike_recognition/train"

# Other parameters
batch_size = 32
num_classes = 2  # Update with the number of classes in your dataset

# Create an ImageDataGenerator with rescaling and other preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Scale pixel values to [0, 1]
    # Add more preprocessing options if needed
)

# Flow the training images from the directory
train_generator = train_datagen.flow_from_directory(
    train_folder_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15
)

# Predict class labels for each image and display if the prediction matches the class
class_names = sorted(os.listdir(train_folder_path))
for i, (images, labels) in enumerate(train_generator):
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    for j in range(len(images)):
        if class_names[predicted_classes[j]] == 'r15':
            image_path = train_generator.filenames[i * batch_size + j]
            img = Image.open(os.path.join(train_folder_path, image_path))
            plt.imshow(img)
            plt.title('Predicted Class: ' + class_names[predicted_classes[j]])
            plt.show()
