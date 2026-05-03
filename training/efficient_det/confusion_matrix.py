import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from test import (
    load_classes,
    find_latest_model,
    get_model_input_size,
    preprocess_frame,
    quantize_input,
    dequantize_output,
    predict_tflite,
    predict_keras,
)


def load_validation_data(data_dir, classes):
    """Carrega todas as imagens de validação e suas labels."""
    images = []
    labels = []
    
    val_dir = Path(data_dir) / 'val'
    
    if not val_dir.exists():
        print(f"Erro: Diretório de validação não encontrado em {val_dir}")
        return [], []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = val_dir / class_name

        if not class_dir.exists():
            print(f"Aviso: Classe '{class_name}' não encontrada em {class_dir}")
            continue

        # Suporta .jpg, .jpeg, .png (minúsculo e maiúsculo)
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in exts:
            image_files.extend(class_dir.glob(ext))
        print(f"Carregando {len(image_files)} imagens da classe '{class_name}'")

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {e}")
    
    return images, labels


def predict_with_model(model, images, model_input_size, use_keras):
    """Faz predições em um conjunto de imagens."""
    predictions = []
    
    for idx, img in enumerate(images):
        if (idx + 1) % 50 == 0:
            print(f"Processando imagem {idx + 1}/{len(images)}")
        
        # Preprocessa a imagem
        processed_img = preprocess_frame(img, model_input_size)
        
        # Faz a predição
        if use_keras:
            output = predict_keras(model, processed_img)
        else:
            output = predict_tflite(model, processed_img)
        
        # Extrai a classe predita
        pred_class = np.argmax(output, axis=-1).flatten()[0]
        predictions.append(pred_class)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Gera matriz de confusão para o modelo de detecção')
    parser.add_argument('--model', type=str, default=None, help='Caminho do modelo (padrão: modelo mais recente)')
    parser.add_argument('--keras', action='store_true', help='Usar modelo Keras ao invés de TFLite')
    parser.add_argument('--data-dir', type=str, default='./data', help='Diretório dos dados')
    parser.add_argument('--classes', type=str, default='./classes.txt', help='Arquivo com lista de classes')
    parser.add_argument('--output', type=str, default=None, help='Arquivo de saída para a imagem')
    parser.add_argument('--report', type=str, default=None, help='Arquivo de saída para o relatório')
    
    args = parser.parse_args()
    
    # Carrega as classes
    print("Carregando classes...")
    classes = load_classes(args.classes)
    if not classes:
        print("Erro: Nenhuma classe encontrada!")
        return
    print(f"Classes encontradas: {classes}")

    # Pasta de relatórios
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Encontra o modelo
    print("\nEncontrando modelo...")
    if args.model is None:
        model_path = find_latest_model('./models', 'keras' if args.keras else 'tflite')
        if model_path is None:
            print("Erro: Nenhum modelo encontrado!")
            return
    else:
        model_path = args.model
    
    print(f"Usando modelo: {model_path}")
    # Nome base do modelo para arquivos
    model_base = Path(model_path).stem
    
    # Carrega o modelo
    print("Carregando modelo...")
    use_keras = args.keras or model_path.endswith('.keras')
    
    if use_keras:
        model = tf.keras.models.load_model(model_path)
        input_shape = model.input_shape
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        model = interpreter
    
    model_input_size = get_model_input_size(input_shape)
    print(f"Tamanho de entrada do modelo: {model_input_size}")
    
    # Carrega dados de validação
    print("\nCarregando dados de validação...")
    images, true_labels = load_validation_data(args.data_dir, classes)
    
    if not images:
        print("Erro: Nenhuma imagem encontrada!")
        return
    
    print(f"Total de imagens carregadas: {len(images)}")
    
    # Faz predições
    print("\nFazendo predições...")
    predicted_labels = predict_with_model(model, images, model_input_size, use_keras)

    # Calcula a matriz de confusão
    print("\nCalculando matriz de confusão...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(classes)))

    # Cria relatório de classificação
    print("\nGerando relatório de classificação...")
    report = classification_report(
        true_labels, predicted_labels,
        labels=range(len(classes)),
        target_names=classes,
        zero_division=0
    )
    print(report)

    # Define nomes dos arquivos de saída
    output_img = args.output or reports_dir / f"confusion_matrix_{model_base}.png"
    output_report = args.report or reports_dir / f"classification_report_{model_base}.txt"

    # Salva o relatório
    with open(output_report, 'w') as f:
        f.write(report)
    print(f"Relatório salvo em: {output_report}")
    
    # Visualiza a matriz de confusão
    print("\nVisualizando matriz de confusão...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Contagem'})
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusão salva em: {output_img}")
    
    # Exibe a matriz de confusão como tabela
    print("\nMatriz de Confusão:")
    print(f"{'Classes':<15}", end='')
    for cls in classes:
        print(f"{cls:<10}", end='')
    print()
    
    for i, cls in enumerate(classes):
        print(f"{cls:<15}", end='')
        for j in range(len(classes)):
            print(f"{cm[i][j]:<10}", end='')
        print()


if __name__ == '__main__':
    main()
