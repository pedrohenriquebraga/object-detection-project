import os
from datetime import datetime
import tensorflow as tf
from keras.applications import EfficientNetB1, EfficientNetB0
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

batch_size = 16
img_size = (320, 320)
epochs = 100 # valor máximo, não total
AUTOTUNE = tf.data.AUTOTUNE
early_stopping_patience = 20
early_stopping_min_delta = 0.001

gpus = tf.config.list_physical_devices('GPU')

base_dir = './data'
train_dir = './data/train'
val_dir = './data/val'
            
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponíveis: {gpus}")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("Nenhuma GPU encontrada. O treinamento será feito na CPU.")

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
    # Center crop para quadrado para evitar distorção
    size = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    offset_height = (tf.shape(image)[0] - size) // 2
    offset_width = (tf.shape(image)[1] - size) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, size, size)
    # Resize para img_size
    image = tf.image.resize(image, img_size)
    return image, label

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

def preprocess(images, labels):
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels

def maybe_grayscale(image, probability=0.25):
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

def augment_single_image(image):
    image = tf.cast(image, tf.float32)
    image = rotation_layer(tf.expand_dims(image, axis=0), training=True)[0]
    image = maybe_grayscale(image)
    image = random_zoom_crop(image)
    return image

def augment_image(images, labels):
    images = tf.map_fn(
        augment_single_image,
        images,
        fn_output_signature=tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
    )
    return images, labels

def calculate_class_weights(dataset, num_classes):
    """
    Calcula pesos das classes de forma inversamente proporcional à frequência.
    Classes com poucas imagens recebem peso maior.
    """
    class_counts = tf.zeros(num_classes)
    
    for _, labels in dataset:
        class_counts = class_counts + tf.reduce_sum(
            tf.one_hot(labels, depth=num_classes), axis=0
        )
    
    total_samples = tf.reduce_sum(class_counts)
    class_weights = total_samples / (num_classes * (class_counts + 1e-7))
    
    return class_weights.numpy()


def representative_dataset():
    for images, _ in calibration_dataset.unbatch().take(100):
        yield [tf.expand_dims(tf.cast(images, tf.float32), axis=0)]


def next_model_path(models_dir, base_name, extension, run_id):
    candidate = os.path.join(models_dir, f"{base_name}_{run_id}.{extension}")
    if not os.path.exists(candidate):
        return candidate

    idx = 1
    while True:
        candidate = os.path.join(models_dir, f"{base_name}_{run_id}_{idx}.{extension}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not check_image_validity(file_path):
            os.remove(file_path)

train_class_names = save_classes_file(base_dir, 'classes.txt')
if not train_class_names:
    raise ValueError('Nenhuma classe encontrada em ./data/train. Verifique a estrutura do dataset.')

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=batch_size)
train_dataset = train_dataset.map(resize_image)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, batch_size=batch_size)
val_dataset = val_dataset.map(resize_image)

rotation_layer = tf.keras.layers.RandomRotation(0.2, fill_mode='reflect')
calibration_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

train_dataset = (
    train_dataset
    .shuffle(1000, reshuffle_each_iteration=True)
    .map(augment_image, num_parallel_calls=AUTOTUNE)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_dataset = val_dataset.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(320, 320, 3))
base_model.trainable = False  # Freeze base model for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)

num_classes = len(train_class_names)
output = Dense(num_classes, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

def sparse_labels_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, depth=num_classes)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

model.compile(optimizer="adam",
              loss=sparse_labels_binary_crossentropy,
              metrics=['accuracy'])

# Calcula pesos das classes para lidar com desbalanceamento
print("\n📊 Calculando pesos das classes...")
class_weights_array = calculate_class_weights(train_dataset, num_classes)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}

print("\n📋 Pesos das classes (para compensar desbalanceamento):")
for i, class_name in enumerate(train_class_names):
    print(f"   {class_name:<15} → peso: {class_weights_dict[i]:.3f}")

os.makedirs('models', exist_ok=True)
run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=early_stopping_min_delta,
    patience=early_stopping_patience,
    mode='min',
    restore_best_weights=True,
    verbose=1,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/efficient_det_20260409-030229.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False
)

try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=[early_stopping_callback, lr_scheduler, checkpoint] # checkpoint aqui
    )
except KeyboardInterrupt:
    print('\nTreinamento interrompido manualmente. Salvando modelo parcial...')
    interrupted_output_path = next_model_path('models', 'efficient_det_interrupted', 'keras', run_id)
    model.save(interrupted_output_path)
    print(f"Modelo parcial salvo em {interrupted_output_path} com sucesso!")
    raise SystemExit(130)

keras_output_path = next_model_path('models', 'efficient_det', 'keras', run_id)
model.save(keras_output_path)
print(f"Modelo salvo em {keras_output_path} com sucesso!")