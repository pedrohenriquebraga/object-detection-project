"""
Script para converter modelo .keras para .tflite.

Por padrao, gera modelo sem Flex Delegate usando apenas TFLITE_BUILTINS.

Uso: python3 convert.py [--quantization float|float16|dynamic|int8] [--runtime builtin|flex|auto] [--input models/efficient_det.keras] [--output models/efficient_det.tflite]
Exemplo: python3 convert.py --quantization dynamic --runtime builtin
"""

import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf


def find_latest_model(models_dir, extension):
    if not os.path.isdir(models_dir):
        return None

    candidates = [
        os.path.join(models_dir, file_name)
        for file_name in os.listdir(models_dir)
        if file_name.endswith(f".{extension}")
    ]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_input_model_path(input_model_path):
    if os.path.exists(input_model_path):
        return input_model_path

    latest_keras = find_latest_model('models', 'keras')
    if latest_keras is not None:
        print(f"Aviso: {input_model_path} nao encontrado. Usando ultimo .keras: {latest_keras}")
        return latest_keras

    raise FileNotFoundError(f"Modelo nao encontrado: {input_model_path}")


def resolve_output_model_path(output_model_path, input_model_path, quantization_mode, runtime_mode):
    output_dir = os.path.dirname(output_model_path) or 'models'
    os.makedirs(output_dir, exist_ok=True)

    if output_model_path.endswith('efficient_det.tflite'):
        input_stem = os.path.splitext(os.path.basename(input_model_path))[0]
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        base_name = f"{input_stem}_{quantization_mode}_{runtime_mode}_{run_id}"
        candidate = os.path.join(output_dir, f"{base_name}.tflite")
    else:
        candidate = output_model_path

    if not os.path.exists(candidate):
        return candidate

    stem, ext = os.path.splitext(candidate)
    idx = 1
    while True:
        indexed_candidate = f"{stem}_{idx}{ext}"
        if not os.path.exists(indexed_candidate):
            return indexed_candidate
        idx += 1

def representative_dataset():
    """
    Dataset representativo para quantização INT8.
    Carrega imagens do dataset de treinamento.
    """
    base_dir = './data'
    train_dir = os.path.join(base_dir, 'train')
    img_size = (512, 512)
    batch_size = 8
    
    if not os.path.exists(train_dir):
        print(f"Aviso: diretório {train_dir} não encontrado. Usando dataset vazio para quantização.")
        return
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size)
    
    def preprocess(images, labels):
        images = tf.keras.applications.efficientnet.preprocess_input(images)
        return images, labels
    
    train_dataset = train_dataset.map(preprocess)
    
    # Usa apenas 100 batches para quantização
    for images, _ in train_dataset.unbatch().take(100):
        yield [tf.expand_dims(tf.cast(images, tf.float32), axis=0)]


def convert_keras_to_tflite(input_model_path, output_model_path, quantization_mode, runtime_mode):
    """
    Converte modelo Keras para TFLite usando concrete functions.
    
    Args:
        input_model_path: str - caminho do modelo .keras
        output_model_path: str - caminho de saída do modelo .tflite
        quantization_mode: str - modo de quantização
    """
    import shutil
    import tempfile
    
    input_model_path = resolve_input_model_path(input_model_path)
    output_model_path = resolve_output_model_path(output_model_path, input_model_path, quantization_mode, runtime_mode)
    
    print(f"Carregando modelo de {input_model_path}...")
    model = tf.keras.models.load_model(input_model_path)
    
    # Tenta converter com o modo preferido, com fallbacks se necessário
    preferred_modes = [quantization_mode]
    
    if quantization_mode == 'int8':
        # INT8 pode falhar, use fallback para dynamic e float
        preferred_modes.extend(['dynamic', 'float'])
    elif quantization_mode == 'float16':
        # float16 pode falhar, usar dynamic e float
        preferred_modes.extend(['dynamic', 'float'])
    
    if runtime_mode == 'builtin':
        runtime_candidates = ['builtin']
    elif runtime_mode == 'flex':
        runtime_candidates = ['flex']
    else:
        runtime_candidates = ['builtin', 'flex']

    tflite_model = None
    selected_mode = None
    selected_runtime = None
    last_error = None

    for runtime_candidate in runtime_candidates:
        for mode in preferred_modes:
            try:
                print(f"Tentando conversão com modo '{mode}' e runtime '{runtime_candidate}'...")

                concrete_func = tf.function(
                    lambda x: model(x, training=False)
                ).get_concrete_function(
                    tf.TensorSpec(
                        shape=[1, 512, 512, 3],
                        dtype=tf.float32
                    )
                )

                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
                converter.experimental_enable_resource_variables = False
                converter.allow_custom_ops = False

                if runtime_candidate == 'builtin':
                    if mode == 'int8':
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    else:
                        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                else:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

                if mode == 'float':
                    pass
                elif mode == 'dynamic':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                elif mode == 'int8':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = representative_dataset
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                elif mode == 'float16':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]

                tflite_model = converter.convert()
                selected_mode = mode
                selected_runtime = runtime_candidate
                print(f"✓ Conversão bem-sucedida com modo '{mode}' e runtime '{runtime_candidate}'")
                break
            except Exception as conversion_error:
                last_error = conversion_error
                print(f"⚠ Falha na conversão com modo '{mode}' e runtime '{runtime_candidate}'.")

        if tflite_model is not None:
            break
    
    if tflite_model is None:
        raise RuntimeError(f"Nenhuma conversão TFLite teve sucesso. Último erro: {last_error}")
    
    # Salva o modelo TFLite
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    
    file_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"\n✓ Modelo convertido com sucesso!")
    print(f"  Arquivo: {output_model_path}")
    print(f"  Tamanho: {file_size_mb:.2f} MB")
    print(f"  Quantização: {selected_mode}")
    print(f"  Runtime TFLite: {selected_runtime}")
    if selected_runtime == 'flex':
        print("  SELECT_TF_OPS: Sim (requer Flex delegate)")
    else:
        print("  SELECT_TF_OPS: Nao (sem Flex delegate)")


def main():
    parser = argparse.ArgumentParser(
        description="Converter modelo Keras para TFLite"
    )
    parser.add_argument(
        '--input',
        default='models/efficient_det.keras',
        help='Caminho do modelo .keras (padrão: models/efficient_det.keras)'
    )
    parser.add_argument(
        '--output',
        default='models/efficient_det.tflite',
        help='Caminho do modelo .tflite (padrão: models/efficient_det.tflite)'
    )
    parser.add_argument(
        '--quantization',
        choices=['int8', 'float16', 'dynamic', 'float'],
        default='dynamic',
        help='Modo de quantização (padrão: dynamic). Para EfficientNet, dynamic ou float são mais estáveis.'
    )
    parser.add_argument(
        '--runtime',
        choices=['builtin', 'flex', 'auto'],
        default='builtin',
        help="Runtime TFLite alvo (padrão: builtin, sem Flex). 'auto' tenta builtin e depois flex."
    )
    args = parser.parse_args()
    
    try:
        convert_keras_to_tflite(args.input, args.output, args.quantization, args.runtime)
    except Exception as e:
        print(f"\n✗ Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
