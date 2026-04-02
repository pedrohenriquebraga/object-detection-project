#!/usr/bin/env python3
"""
Script para analisar viés do modelo.
Mostra:
- Distribuição de predições por classe real
- Taxa de acerto por classe (recall)
- Matriz de confusão simplificada
- Classes mais preditas (indicador de viés)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt

def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_latest_model(use_keras=True):
    models_dir = './models'
    if use_keras:
        candidates = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    else:
        candidates = [f for f in os.listdir(models_dir) if f.endswith('.tflite') and '_builtin_' in f]
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    return os.path.join(models_dir, candidates[0])

def analyze_model_bias(model_path, val_dir, classes, use_keras=True):
    """Analisa as predições do modelo no dataset de validação"""
    
    if use_keras:
        model = tf.keras.models.load_model(model_path)
    else:
        from test import get_model_input_size, quantize_input, dequantize_output
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_size = get_model_input_size(interpreter.get_input_details()[0]['shape'])
    
    img_size = (320, 320)
    
    # Carrega dataset de validação
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=32,
        shuffle=False,
        class_names=classes
    )
    
    # Estatísticas
    predictions_by_class = defaultdict(list)  # true_label -> [predictions]
    confusion_matrix = {c: defaultdict(int) for c in classes}
    total_samples = 0
    correct_samples = 0
    
    for images, true_labels in val_dataset:
        images = tf.keras.applications.efficientnet.preprocess_input(images)
        
        if use_keras:
            logits = model.predict(images, verbose=0)
        else:
            # TFLite inference
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logits = []
            for img in images:
                img_array = np.expand_dims(img, axis=0)
                quantized = quantize_input(img_array, input_details)
                interpreter.set_tensor(input_details[0]['index'], quantized)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                logits.append(dequantize_output(output, output_details)[0])
            logits = np.array(logits)
        
        predicted_classes = np.argmax(logits, axis=1)
        
        for true_label, pred_class in zip(true_labels.numpy(), predicted_classes):
            true_class_name = classes[true_label]
            pred_class_name = classes[pred_class]
            
            predictions_by_class[true_class_name].append(pred_class_name)
            confusion_matrix[true_class_name][pred_class_name] += 1
            
            if true_label == pred_class:
                correct_samples += 1
            total_samples += 1
    
    return predictions_by_class, confusion_matrix, correct_samples, total_samples

def main():
    parser = argparse.ArgumentParser(description="Analisa viés do modelo EfficientDet")
    parser.add_argument('--keras', action='store_true', help='Usar modelo Keras (padrão)')
    parser.add_argument('--tflite', action='store_true', help='Usar modelo TFLite')
    parser.add_argument('--model', type=str, default=None, help='Caminho do modelo (opcional)')
    args = parser.parse_args()
    
    if not args.keras and not args.tflite:
        args.keras = True
    
    classes = load_classes('./classes.txt')
    model_path = args.model or get_latest_model(use_keras=args.keras)
    
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    print(f"\n📊 Analisando modelo: {os.path.basename(model_path)}")
    print("="*70)
    
    predictions_by_class, confusion_matrix, correct, total = analyze_model_bias(
        model_path, './data/val', classes, use_keras=args.keras
    )
    
    accuracy = correct / total * 100
    
    # Mostra análise por classe
    print(f"\n📈 ANÁLISE DE ACURÁCIA POR CLASSE (Recall):")
    print(f"{'Classe':<15} {'Acertos':<10} {'Total':<10} {'Recall':<10} {'Confiança':<15}")
    print("-"*70)
    
    recall_by_class = {}
    for class_name in sorted(classes):
        if class_name in predictions_by_class:
            predictions = predictions_by_class[class_name]
            correct_count = predictions.count(class_name)
            total_count = len(predictions)
            recall = correct_count / total_count * 100 if total_count > 0 else 0
            recall_by_class[class_name] = recall
            
            print(f"{class_name:<15} {correct_count:<10} {total_count:<10} {recall:>6.1f}%     ", end="")
            
            if recall < 50:
                print("🔴 CRÍTICO")
            elif recall < 70:
                print("🟡 BAIXO")
            else:
                print("🟢 OK")
    
    print("-"*70)
    print(f"{'TOTAL':<15} {correct:<10} {total:<10} {accuracy:>6.1f}%")
    
    # Analisa viés - qual classe é mais predita?
    print(f"\n🎯 ANÁLISE DE VIÉS (Qual classe é mais predita?):")
    print(f"{'Classe':<15} {'Frequência Predita':<25} {'Taxa de Viés'}")
    print("-"*70)
    
    all_predictions = []
    for preds_list in predictions_by_class.values():
        all_predictions.extend(preds_list)
    
    if all_predictions:
        from collections import Counter
        prediction_counts = Counter(all_predictions)
        
        for class_name in sorted(classes):
            count = prediction_counts.get(class_name, 0)
            proportion = count / len(all_predictions) * 100
            expected = (100 / len(classes))
            bias = proportion - expected
            
            print(f"{class_name:<15} {count:<25} ", end="")
            
            if bias > 20:
                print(f"🔴 +{bias:.1f}% (VIÉS MUITO ALTO)")
            elif bias > 5:
                print(f"🟡 +{bias:.1f}% (VIÉS)")
            elif bias < -10:
                print(f"🟠 {bias:.1f}% (SUB-PREDITA)")
            else:
                print(f"🟢 {bias:.1f}% (BALANCEADO)")
    
    # Mostra matriz de confusão para classe que é mais predita
    if prediction_counts:
        most_predicted = prediction_counts.most_common(1)[0][0]
        print(f"\n🔍 DETALHES DA CLASSE MAIS PREDITA: '{most_predicted}'")
        print(f"   Predita {prediction_counts[most_predicted]}x em {total} predições ({prediction_counts[most_predicted]/total*100:.1f}%)")
        print(f"   Verdadeiros positivos (acertos): {confusion_matrix[most_predicted][most_predicted]}")
        
        false_positives = sum(
            v for class_name, counts in confusion_matrix.items()
            if class_name != most_predicted
            for k, v in counts.items()
            if k == most_predicted
        )
        print(f"   Falsos positivos (erros): {false_positives}")

if __name__ == "__main__":
    main()
