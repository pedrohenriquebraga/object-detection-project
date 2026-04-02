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


def predict_tflite(interpreter, img_array):
	input_details = interpreter.get_input_details()
	input_dtype = input_details[0]['dtype']
	input_tensor = img_array.astype(input_dtype)
	if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][0] == 1:
		interpreter.set_tensor(input_details[0]['index'], input_tensor)
	else:
		interpreter.set_tensor(input_details[0]['index'], np.resize(input_tensor, input_details[0]['shape']))
	interpreter.invoke()
	return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


def predict_keras(model, img_array):
	return model.predict(img_array, verbose=0)


def draw_label(frame, label):
	cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return frame


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="EfficientDet real-time classification (Keras or TFLite)")
	parser.add_argument('--keras', action='store_true', help='Use Keras model for real-time inference')
	parser.add_argument('--tflite', action='store_true', help='Use TFLite model for real-time inference')
	parser.add_argument('--model-path', type=str, default=None, help='Path to the model file')
	parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
	args = parser.parse_args()

	classes = load_classes('./classes.txt')
	if not classes:
		classes = ['bus', 'cars', 'cats', 'chairs', 'dogs', 'doors']

	if args.keras and args.tflite:
		print('Escolha apenas um backend: --keras ou --tflite.')
		exit(1)

	if not args.keras and not args.tflite:
		if args.model_path and args.model_path.endswith('.keras'):
			args.keras = True
		else:
			args.tflite = True

	model_path = args.model_path
	if not model_path:
		model_path = './models/efficient_det.keras' if args.keras else './models/efficient_det.tflite'

	if args.keras:
		model = tf.keras.models.load_model(model_path)
		input_size = get_model_input_size(model.input_shape)
	else:
		interpreter = tf.lite.Interpreter(model_path=model_path)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		input_size = get_model_input_size(input_details[0]['shape'])

	cap = cv2.VideoCapture(2)
	if not cap.isOpened():
		print('Não foi possível abrir a câmera.')
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
		class_name = classes[pred_class + 1] if pred_class < len(classes) else f'class_{pred_class}'
  
		if (conf >= 0.6):
			label = f"{class_name}: {conf * 100:.1f}%"
			frame = draw_label(frame, label)
			print(label)

		cv2.imshow('EfficientDet Real-Time', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
