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

batch_size = 4
img_size = (512, 512)
epochs = 30
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size)
train_dataset = train_dataset.map(resize_image)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=batch_size)
val_dataset = val_dataset.map(resize_image)

rotation_layer = tf.keras.layers.RandomRotation(0.2, fill_mode='reflect')

def preprocess(images, labels):
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels


def maybe_grayscale(image, probability=0.35):
    return tf.cond(
        tf.random.uniform([]) < probability,
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)),
        lambda: image,
    )


def random_zoom_crop(image, min_scale=0.75, max_scale=0.95):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]

    scale = tf.random.uniform([], min_scale, max_scale)
    crop_height = tf.maximum(1, tf.cast(tf.cast(height, tf.float32) * scale, tf.int32))
    crop_width = tf.maximum(1, tf.cast(tf.cast(width, tf.float32) * scale, tf.int32))
    offset_height = tf.random.uniform([], 0, height - crop_height + 1, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, width - crop_width + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    image = tf.image.resize(image, img_size)
    image.set_shape([img_size[0], img_size[1], 3])
    return image


def random_cutout(image, min_fraction=0.12, max_fraction=0.35):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]

    cutout_fraction = tf.random.uniform([], min_fraction, max_fraction)
    cutout_height = tf.maximum(1, tf.cast(tf.cast(height, tf.float32) * cutout_fraction, tf.int32))
    cutout_width = tf.maximum(1, tf.cast(tf.cast(width, tf.float32) * cutout_fraction, tf.int32))
    offset_height = tf.random.uniform([], 0, height - cutout_height + 1, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, width - cutout_width + 1, dtype=tf.int32)

    cutout_mask = tf.ones([cutout_height, cutout_width, channels], dtype=image.dtype)
    cutout_mask = tf.pad(
        cutout_mask,
        [[offset_height, height - offset_height - cutout_height],
         [offset_width, width - offset_width - cutout_width],
         [0, 0]],
        constant_values=0,
    )
    return image * (1.0 - cutout_mask)


def augment_single_image(image):
    image = tf.cast(image, tf.float32)
    image = rotation_layer(tf.expand_dims(image, axis=0), training=True)[0]
    image = maybe_grayscale(image)
    image = random_zoom_crop(image)
    image = random_cutout(image)
    return image


def augment_image(images, labels):
    images = tf.map_fn(
        augment_single_image,
        images,
        fn_output_signature=tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
    )
    return images, labels


calibration_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

train_dataset = (
    train_dataset
    .shuffle(1000, reshuffle_each_iteration=True)
    .map(augment_image, num_parallel_calls=AUTOTUNE)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_dataset = val_dataset.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def representative_dataset():
    for images, _ in calibration_dataset.unbatch().take(100):
        yield [tf.expand_dims(tf.cast(images, tf.float32), axis=0)]

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
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/efficient_det.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[checkpoint]
)

model.save('models/efficient_det.keras')
print("Modelo salvo em models/efficient_det.keras com sucesso!")

export_tflite = True
if export_tflite:
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with open("models/efficient_det.tflite", "wb") as f:
            f.write(tflite_model)
        print("Modelo convertido para models/efficient_det.tflite com sucesso!")
    except Exception as e:
        print(f"Aviso: falha ao converter para TFLite: {e}")
else:
    print("Exportação TFLite desativada. Defina para gerar models/efficient_det.tflite.")
