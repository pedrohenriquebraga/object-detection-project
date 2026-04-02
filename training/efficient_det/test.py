import argparse
import os

import cv2
import numpy as np
import tensorflow as tf


def load_classes(classes_path):
	if not os.path.exists(classes_path):
		return []

	with open(classes_path, 'r') as f:
		return [line.strip() for line in f if line.strip()]


def find_latest_model(models_dir, extension, contains=None):
	if not os.path.isdir(models_dir):
		return None

	candidates = []
	for file_name in os.listdir(models_dir):
		if not file_name.endswith(f'.{extension}'):
			continue
		if contains and contains not in file_name:
			continue
		candidates.append(os.path.join(models_dir, file_name))

	if not candidates:
		return None

	candidates.sort(key=os.path.getmtime, reverse=True)
	return candidates[0]


def get_default_model_path(use_keras):
	if use_keras:
		return find_latest_model('./models', 'keras') or './models/efficient_det.keras'

	builtin_model = find_latest_model('./models', 'tflite', contains='_builtin_')
	if builtin_model is not None:
		return builtin_model

	return find_latest_model('./models', 'tflite') or './models/efficient_det.tflite'


def get_model_input_size(model_input_shape, fallback=(512, 512)):
	if len(model_input_shape) >= 3:
		height = model_input_shape[1] or fallback[0]
		width = model_input_shape[2] or fallback[1]
		return int(height), int(width)
	return fallback


def preprocess_frame(frame, target_size):
	img = cv2.resize(frame, target_size)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
	img = np.expand_dims(img, axis=0)
	return img


def quantize_input(img_array, input_details):
	input_dtype = input_details[0]['dtype']
	if input_dtype == np.int8 or input_dtype == np.uint8:
		scale, zero_point = input_details[0]['quantization']
		if scale == 0:
			return img_array.astype(input_dtype)
		quantized = np.round(img_array / scale + zero_point)
		info = np.iinfo(input_dtype)
		return np.clip(quantized, info.min, info.max).astype(input_dtype)
	return img_array.astype(input_dtype)


def dequantize_output(output_array, output_details):
	output_dtype = output_details[0]['dtype']
	if output_dtype == np.int8 or output_dtype == np.uint8:
		scale, zero_point = output_details[0]['quantization']
		if scale == 0:
			return output_array.astype(np.float32)
		return (output_array.astype(np.float32) - zero_point) * scale
	return output_array.astype(np.float32)


def predict_tflite(interpreter, img_array):
	input_details = interpreter.get_input_details()
	input_tensor = quantize_input(img_array, input_details)
	if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][0] == 1:
		interpreter.set_tensor(input_details[0]['index'], input_tensor)
	else:
		interpreter.set_tensor(input_details[0]['index'], np.resize(input_tensor, input_details[0]['shape']))
	interpreter.invoke()
	output_details = interpreter.get_output_details()
	output_tensor = interpreter.get_tensor(output_details[0]['index'])
	return dequantize_output(output_tensor, output_details)


def predict_keras(model, img_array):
	return model.predict(img_array, verbose=0)


def draw_label(frame, label):
	cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="EfficientDet real-time classification (Keras or TFLite)")
	parser.add_argument('--keras', action='store_true', help='Use Keras model for real-time inference')
	parser.add_argument('--tflite', action='store_true', help='Use TFLite model for real-time inference')
	parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
	parser.add_argument('--model', type=str, default=None, help='Custom model path (optional)')
	args = parser.parse_args()

	classes = load_classes('./classes.txt')

	if args.keras and args.tflite:
		print('Escolha apenas um backend: --keras ou --tflite.')
		exit(1)

	default_model_path = get_default_model_path(args.keras)
	# Se nenhum backend for informado, usa TFLite por padrão.
	if not args.keras and not args.tflite:
		args.tflite = True

	model_path = args.model or default_model_path
	if not os.path.exists(model_path):
		print(f"Modelo não encontrado: {model_path}")
		exit(1)

	if args.keras:
		model = tf.keras.models.load_model(model_path)
		input_size = get_model_input_size(model.input_shape)
	else:
		try:
			# Carrega interpretador TFLite
			print(f"Carregando modelo TFLite: {model_path}")
			interpreter = tf.lite.Interpreter(model_path=model_path)
			interpreter.allocate_tensors()
			print("✓ Modelo TFLite carregado com sucesso!")
		except RuntimeError as e:
			if "Select TensorFlow op" in str(e):
				print("\n⚠️  AVISO: O modelo TFLite atual usa SELECT_TF_OPS (Flex Delegate).")
				print("   Para rodar sem Flex, reconverta com runtime builtin:")
				print("   ./convert.sh dynamic builtin")
				print("   ou")
				print("   python3 convert.py --quantization dynamic --runtime builtin")
				exit(1)
			else:
				raise e
		
		input_details = interpreter.get_input_details()
		input_size = get_model_input_size(input_details[0]['shape'])

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		print(f'Não foi possível abrir a câmera no índice {args.camera}.')
		exit(1)

	print("Pressione 'q' para sair.")
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		img_array = preprocess_frame(frame, input_size)
		if args.keras:
			predictions = predict_keras(model, img_array)
		else:
			predictions = predict_tflite(interpreter, img_array)

		predictions = np.squeeze(predictions)
		pred_class = int(np.argmax(predictions))
		conf = float(np.max(predictions))
		class_name = classes[pred_class] if pred_class < len(classes) else f'class_{pred_class}'
  
		if (conf >= 0.75):
			label = f"{class_name}: {conf * 100:.1f}%"
			frame = draw_label(frame, label)
			print(label)

		cv2.imshow('EfficientDet Real-Time', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
