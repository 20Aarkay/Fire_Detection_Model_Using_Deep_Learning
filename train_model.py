import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
base_dir = 'Dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
model_dir = 'fire detection model'
model_path = os.path.join(model_dir, 'fire_detection_model.keras')

os.makedirs(model_dir, exist_ok=True)

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 15

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

import json
class_indices_path = os.path.join(model_dir, 'class_indices.json')
with open(class_indices_path, 'w') as f:
    json.dump(train_gen.class_indices, f)
print(f"✅ Class indices saved to: {class_indices_path}")

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: fire, none fire, smoke
])

model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]
)

# Evaluation
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save(model_path)
print(f"✅ Model saved at: {model_path}")
