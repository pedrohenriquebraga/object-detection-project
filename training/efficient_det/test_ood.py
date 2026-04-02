#!/usr/bin/env python3
"""
Script melhorado de teste com detecção de Out-of-Distribution (OOD).
Alerta quando a imagem está fora da distribuição do modelo.
"""

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from test import load_classes, get_model_input_size, preprocess_frame, predict_keras

def analyze_confidence(probabilities, threshold=0.6, ood_threshold=0.5):
    """
    Analisa a confiança da predição.
    
    - Confiança baixa (<50%): OOD provável (imagem fora da distribuição)
    - Confiança média (50-60%): Predição incerta
    - Confiança alta (>60%): Confiável
    """
    max_confidence = np.max(probabilities)
    top_2_indices = np.argsort(probabilities)[::-1][:2]
    
    # Verifica se é ambíguo (duas classes com similar probabilidade)
    if len(probabilities) > 1:
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        # Max entropy para N classes é log(N), normalized: 0-1
        max_entropy = np.log(len(probabilities))
        entropy_score = entropy / max_entropy
    else:
        entropy_score = 0
    
    is_ood = max_confidence < ood_threshold
    is_uncertain = entropy_score > 0.7
    
    return {
        'confidence': max_confidence,
        'entropy_score': entropy_score,
        'is_ood': is_ood,
        'is_uncertain': is_uncertain,
        'top_2': top_2_indices
    }

def print_prediction(class_name, classes, probabilities, stats):
    """Printa predição com indicadores visuais"""
    confidence = stats['confidence']
    
    # Cor visual baseada em confiança
    if confidence < 0.3:
        marker = "🔴 MUITO BAIXA - PROVÁVEL OOD"
    elif confidence < 0.5:
        marker = "🟠 BAIXA - POSSÍVEL OOD"
    elif confidence < 0.7:
        marker = "🟡 MÉDIA - INCERTA"
    else:
        marker = "🟢 ALTA - CONFIÁVEL"
    
    print(f"   Predição: {class_name:<15} {confidence*100:>5.1f}%  {marker}")
    
    if stats['is_ood']:
        print(f"   ⚠️  AVISO: Imagem pode estar FORA DA DISTRIBUIÇÃO (OOD)")
        print(f"      A classe '{class_name}' pode não ser correta!")
    
    if stats['is_uncertain']:
        print(f"   ℹ️  Predição ambígua - múltiplas classes similares:")
        top_3 = np.argsort(probabilities)[::-1][:3]
        for idx in top_3:
            print(f"      - {classes[idx]:<15} {probabilities[idx]*100:>5.1f}%")

def test_image(image_path, model, classes, input_size, confidence_threshold=0.6):
    """Testa uma imagem individual"""
    
    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        return False
    
    try:
        img = tf.io.read_file(image_path)
        frame = tf.image.decode_image(img).numpy()
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {e}")
        return False
    
    # Processa
    processed = preprocess_frame(frame, input_size)
    
    # Predição
    output = predict_keras(model, processed)
    probabilities = output[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = classes[predicted_class_idx]
    
    # Análise
    stats = analyze_confidence(probabilities)
    
    print(f"\n📸 Testando: {os.path.basename(image_path)}")
    print(f"   Tamanho original: {frame.shape}")
    print(f"   Tamanho processado: {input_size}")
    print("-" * 60)
    
    print_prediction(predicted_class, classes, probabilities, stats)
    
    if stats['confidence'] < confidence_threshold:
        print(f"\n⚠️  CONFIANÇA BAIXA (< {confidence_threshold*100:.0f}%)")
        print(f"   Considere:")
        print(f"   1. Usar threshold de confiança mínimo")
        print(f"   2. Treinar modelo com mais dados desta categoria")
        print(f"   3. Usar modelo de object detection se há múltiplos objetos")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Teste avançado com detecção OOD")
    parser.add_argument('--image', type=str, default=None, help='Caminho da imagem a testar')
    parser.add_argument('--dir', type=str, default='./test_images', help='Diretório com imagens de teste')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold de confiança (0-1)')
    parser.add_argument('--keras', action='store_true', help='Usar modelo Keras (padrão)')
    args = parser.parse_args()
    
    classes = load_classes('./classes.txt')
    
    print("\n" + "="*70)
    print("TESTE AVANÇADO COM DETECÇÃO OOD")
    print("="*70)
    print(f"Classes aprendidas: {', '.join(classes)}")
    print(f"Threshold de confiança: {args.threshold*100:.0f}%")
    print("="*70)
    
    model = tf.keras.models.load_model('./models/efficient_det_20260402-214346.keras')
    input_size = get_model_input_size(model.input_shape)
    
    if args.image:
        # Testa imagem específica
        test_image(args.image, model, classes, input_size, args.threshold)
    else:
        # Testa todas as imagens do diretório
        if not os.path.isdir(args.dir):
            print(f"❌ Diretório não encontrado: {args.dir}")
            return
        
        for img_name in sorted(os.listdir(args.dir)):
            img_path = os.path.join(args.dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            test_image(img_path, model, classes, input_size, args.threshold)
    
    print("\n" + "="*70)
    print("RESUMO:")
    print("  🔴 MUITO BAIXA (0-30%)     : Rejeitar predição")
    print("  🟠 BAIXA (30-50%)          : Possível OOD, tentar outras classes")
    print("  🟡 MÉDIA (50-70%)          : Usar com cuidado, considerar threshold")
    print("  🟢 ALTA (70%+)             : Confiável")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
