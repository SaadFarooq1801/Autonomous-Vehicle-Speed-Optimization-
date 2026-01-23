import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout   
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

img_size = 32  
batch_size = 32
epochs = 20  #Number of times the model sees the dataset
learning_rate = 0.001

def load_data(data_path):
    images = []
    labels = []
    classes = os.listdir(data_path)
    num_classes = len(classes)
    print(f"Found {num_classes} classes")
    
    for class_num, class_folder in enumerate(classes):
        folder_path = os.path.join(data_path, class_folder)
        if not os.path.isdir(folder_path):
            continue
            
        image_files = os.listdir(folder_path)
        print(f"Loading class {class_folder}: {len(image_files)} images")
        
        for image_file in image_files:
            try:
                image_path = os.path.join(folder_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading {image_file}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, num_classes

X_train, y_train, num_classes = load_data('dataset/Train')
X_test, y_test, _ = load_data('dataset/Test')
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

def preprocess_images(images):
    images = images / 255.0 
    return images

X_train = preprocess_images(X_train)
X_test = preprocess_images(X_test)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")

datagen = ImageDataGenerator(
    rotation_range=10,      # Randomly rotate images by ±10 degrees
    zoom_range=0.15,        # Randomly zoom in/out by 15%
    width_shift_range=0.1,  # Shift horizontally by 10%
    height_shift_range=0.1, # Shift vertically by 10%
    shear_range=0.1,        # Apply shear transformation
    brightness_range=[0.8, 1.2]  # Vary brightness
)
datagen.fit(X_train)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) 
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_model()
model.summary()
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // batch_size,
    verbose=1
)
model.save('model_trained.h5')
print("Model saved successfully!")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

predictions = model.predict(X_test[:5])
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    actual_class = np.argmax(y_test[i])
    confidence = pred[predicted_class] * 100
    print(f"Image {i+1}: Predicted={predicted_class}, Actual={actual_class}, Confidence={confidence:.2f}%")

def predict_speed_sign(image_path, model, class_names):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  #batch dimension
    
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    
    print(f"Predicted Speed Limit: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    
    return class_names[predicted_class], confidence

class_names = sorted(os.listdir('dataset/Train'))
predicted_speed, confidence = predict_speed_sign('test_image.jpg', model, class_names)
