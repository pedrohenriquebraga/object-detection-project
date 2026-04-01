import argparse
import numpy as np
import tensorflow as tf
import cv2

def predict_tflite(interpreter, img_array):
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
	interpreter.invoke()
	# Se houver múltiplas saídas, colete todas
	if len(output_details) == 3:
		boxes = interpreter.get_tensor(output_details[0]['index'])
		classes = interpreter.get_tensor(output_details[1]['index'])
		scores = interpreter.get_tensor(output_details[2]['index'])
		return boxes, classes, scores
	else:
		# fallback: retorna só a predição (ex: classificação)
		predictions = interpreter.get_tensor(output_details[0]['index'])
		return predictions

def preprocess_frame(frame):
	img = cv2.resize(frame, (320, 320))
	img = img.astype(np.float32)
	img = np.expand_dims(img, axis=0)
	return img

def draw_boxes(frame, boxes, classes, scores, class_names, threshold=0.3):
	h, w, _ = frame.shape
	for i in range(len(scores)):
		if scores[i] > threshold:
			ymin, xmin, ymax, xmax = boxes[i]
			left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
			label = f"{class_names[int(classes[i])]}: {scores[i]*100:.1f}%"
			cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	return frame

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="EfficientDet real-time detection (TFLite)")
	parser.add_argument('--tflite', action='store_true', help='Use TFLite model (required for real-time)')
	parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
	args = parser.parse_args()

	classes = ['carro', "gato", "cadeira", "cachorro", "porta"]

	if not args.tflite:
		print("Apenas o modo TFLite é suportado para detecção em tempo real.")
		exit(1)

	interpreter = tf.lite.Interpreter(model_path="./models/efficient_det.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		print("Não foi possível abrir a câmera.")
		exit(1)

	print("Pressione 'q' para sair.")
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		img_array = preprocess_frame(frame)
		predictions = predict_tflite(interpreter, img_array)

		# Se predictions for uma tupla de 3, desenha boxes
		if isinstance(predictions, tuple) and len(predictions) == 3:
			boxes, classes_pred, scores = predictions
			print("boxes:", boxes)
			print("classes_pred:", classes_pred)
			print("scores:", scores)
			frame = draw_boxes(frame, boxes[0], classes_pred[0], scores[0], classes)
		else:
			# fallback: só classificação
			print("predictions:", predictions)
			pred_class = np.argmax(predictions)
			conf = np.max(predictions)
			label = f"{classes[pred_class]}: {conf*100:.1f}%"
			cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

		cv2.imshow('EfficientDet Real-Time', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
