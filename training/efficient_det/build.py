import os
from datetime import datetime
import tensorflow as tf
from keras.applications import EfficientNetB1, EfficientNetB0
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from augmentations import (
    build_augment_image,
    preprocess,
)

batch_size = 8
img_size = (320, 320)
epochs = 100 # valor máximo, não total
# Staged training: primeiro congelamos o backbone, depois descongelamos para fine-tune
epochs_frozen = 10
AUTOTUNE = tf.data.AUTOTUNE
early_stopping_patience = 7
early_stopping_min_delta = 0.001

# Optimization / regularization
base_lr = 1e-4
fine_tune_lr = 1e-5
weight_decay = 1e-4
dropout_rate = 0.5
# Shuffle buffer reduzido para economizar RAM. Ajuste via EFFICIENT_DET_SHUFFLE_BUFFER env var.
shuffle_buffer = int(os.environ.get('EFFICIENT_DET_SHUFFLE_BUFFER', '64'))

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

    print(f"classes.txt atualizado com {len(all_classes)} classes encontradas em {base_dir}.")
    return train_classes

def resize_image(image, label):
    image = tf.image.resize_with_pad(image, img_size[0], img_size[1])
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


def write_model_info_txt(model_path, metrics, base_model_name, classes_count, logs_dir='logs'):
    os.makedirs(logs_dir, exist_ok=True)

    model_file_name = os.path.basename(model_path)
    model_stem, _ = os.path.splitext(model_file_name)
    info_path = os.path.join(logs_dir, f"{model_stem}.txt")

    def metric_value(name):
        value = metrics.get(name)
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)

    with open(info_path, 'w') as f:
        f.write(f"model_file={model_file_name}\n")
        f.write(f"base_model={base_model_name}\n")
        f.write(f"classes_count={classes_count}\n")
        f.write(f"accuracy={metric_value('accuracy')}\n")
        f.write(f"val_accuracy={metric_value('val_accuracy')}\n")
        f.write(f"loss={metric_value('loss')}\n")
        f.write(f"val_loss={metric_value('val_loss')}\n")

    print(f"Informacoes do modelo salvas em {info_path}")

for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not check_image_validity(file_path):
            os.remove(file_path)

train_class_names = save_classes_file(base_dir, 'classes.txt')
if not train_class_names:
    raise ValueError('Nenhuma classe encontrada em ./data/train. Verifique a estrutura do dataset.')

num_classes = len(train_class_names)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, batch_size=batch_size, label_mode='categorical')
train_dataset = train_dataset.map(resize_image)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, batch_size=batch_size, label_mode='categorical')
val_dataset = val_dataset.map(resize_image)

rotation_layer = tf.keras.layers.RandomRotation(0.1, fill_mode='reflect')
calibration_dataset = train_dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
augment_image = build_augment_image(img_size, rotation_layer)


train_dataset = (
    train_dataset
    .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    .map(augment_image, num_parallel_calls=AUTOTUNE)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(320, 320, 3))
# start with the backbone frozen for stable transfer learning
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(dropout_rate)(x)

# Use softmax + CategoricalCrossentropy for single-label multiclass classification
output = Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
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

# Save best model during training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('models', f'best_{run_id}.keras'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
)

log_dir = "logs/fit/" + run_id
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

try:
    history = None
    # Phase 1: train top (backbone frozen)
    if epochs_frozen > 0:
        history_phase1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=min(epochs_frozen, epochs),
            callbacks=[early_stopping_callback, lr_scheduler, tensorboard_callback, checkpoint_callback]
        )
        history = history_phase1

    # Phase 2: unfreeze backbone and fine-tune
    if epochs > epochs_frozen:
        base_model.trainable = True
        # recompile with lower LR for fine-tuning
        optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
        model.compile(optimizer=optimizer_finetune,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

        history_phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            initial_epoch=(history.history['loss'].__len__() if history and hasattr(history, 'history') else 0),
            epochs=epochs,
            callbacks=[early_stopping_callback, lr_scheduler, tensorboard_callback, checkpoint_callback]
        )

        # merge histories
        if history and hasattr(history, 'history'):
            for k, v in history_phase2.history.items():
                history.history.setdefault(k, []).extend(v)
        else:
            history = history_phase2
    
except KeyboardInterrupt:
    print('\nTreinamento interrompido manualmente. Salvando modelo parcial...')
    interrupted_output_path = next_model_path('models', 'efficient_det_interrupted', 'keras', run_id)
    model.save(interrupted_output_path)
    write_model_info_txt(
        model_path=interrupted_output_path,
        metrics={},
        base_model_name=base_model.name,
        classes_count=num_classes,
        logs_dir='logs',
    )
    print(f"Modelo parcial salvo em {interrupted_output_path} com sucesso!")
    raise SystemExit(130)

keras_output_path = next_model_path('models', 'efficient_det', 'keras', run_id)
model.save(keras_output_path)
final_metrics = {}
if 'history' in locals() and hasattr(history, 'history'):
    for metric_name, metric_values in history.history.items():
        if metric_values:
            final_metrics[metric_name] = metric_values[-1]

write_model_info_txt(
    model_path=keras_output_path,
    metrics=final_metrics,
    base_model_name=base_model.name,
    classes_count=num_classes,
    logs_dir='logs',
)
print(f"Modelo salvo em {keras_output_path} com sucesso!")