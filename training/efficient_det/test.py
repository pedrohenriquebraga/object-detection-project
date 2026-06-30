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



