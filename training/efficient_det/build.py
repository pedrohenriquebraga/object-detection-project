from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import os
import tensorflow as tf


def list_class_names(directory):
    if not os.path.isdir(directory):
        return []

    return sorted(
        entry
        for entry in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, entry)) and not entry.startswith('.')
    )


def save_classes_file(base_dir, output_file):
    train_classes = list_class_names(os.path.join(base_dir, 'train'))
    val_classes = list_class_names(os.path.join(base_dir, 'val'))
    all_classes = sorted(set(train_classes) | set(val_classes))

    with open(output_file, 'w') as f:
        for class_name in all_classes:
            f.write(f"{class_name}\n")

    if set(train_classes) != set(val_classes):
        print('Aviso: classes diferentes entre train e val. classes.txt foi salvo com a uniao das classes encontradas.')

    print(f"classes.txt atualizado com {len(all_classes)} classes encontradas em {base_dir}.")
    print(f"Numero de classes usadas no treino (data/train): {len(train_classes)}")
    return train_classes

def resize_image(image, label):
    image = tf.image.resize(image, (512, 512))
    return image, label

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponíveis: {gpus}")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("Nenhuma GPU encontrada. O treinamento será feito na CPU.")
    
    
def check_image_validity(image_path):
    try:
        img = tf.io.read_file(image_path)
        decoded_img = tf.image.decode_image(img)
        if decoded_img.shape[-1] not in [1, 3, 4]:
            print(f"Invalid image: {image_path}")
            return False
        return True
    except Exception as e:
        print(f"Error with image {image_path}: {e}")
        return False

# Validate images in the dataset
base_dir = './data'
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not check_image_validity(file_path):
            os.remove(file_path)

train_dir = './data/train'
val_dir = './data/val'

train_class_names = save_classes_file(base_dir, 'classes.txt')
if not train_class_names:
    raise ValueError('Nenhuma classe encontrada em ./data/train. Verifique a estrutura do dataset.')

batch_size = 32
img_size = (512, 512)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size)
train_dataset = train_dataset.map(resize_image)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size)
val_dataset = val_dataset.map(resize_image)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

def preprocess(images, labels):
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
base_model.trainable = False  # Freeze base model for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)

num_classes = len(train_class_names)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

os.makedirs('models', exist_ok=True)
# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath='models/efficient_det.keras',
#     monitor='val_accuracy',
#     save_best_only=True,
#     save_weights_only=False,
# )

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    # callbacks=[checkpoint]
)

model.save('models/efficient_det.keras')
print("Modelo salvo em models/efficient_det.keras com sucesso!")

export_tflite = os.environ.get('EXPORT_TFLITE', '0') == '1'
if export_tflite:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open("models/efficient_det.tflite", "wb") as f:
            f.write(tflite_model)
        print("Modelo convertido para models/efficient_det.tflite com sucesso!")
    except Exception as e:
        print(f"Aviso: falha ao converter para TFLite: {e}")
else:
    print("Exportação TFLite desativada. Defina EXPORT_TFLITE=1 para gerar models/efficient_det.tflite.")
