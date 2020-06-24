# użycie
# python3 video-text-detect.py --east frozen_east_text_detection.pb

# import paczek
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
# import pytesseract 
import pytesseract


def decode_predictions(scores, geometry):
	# pobierz liczbę wierszy i kolumn z objętości wyników, a następnie zainicjalizuj zestaw ramek i odpowiadające im wyniki ufności
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loopuj po wierszach tablicy
	for y in range(0, numRows):
		# wyodrębnij wyniki (prawdopodobieństwa), a następnie dane geometryczne wykorzystane do uzyskania potencjalnych współrzędnych ramki granicznej otaczających tekst
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loopuj po kolumnach tablicy
		for x in range(0, numCols):
			# jeśli nasz wynik nie ma wystarczającego prawdopodobieństwa, zignoruj go
			if scoresData[x] < args["min_confidence"]:
				continue

			# oblicz współczynnik przesunięcia jako naszą wynikową funkcję - mapy będą 4x mniejsze niż obraz wejściowy
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# użyj geometrii, aby wyliczyć szerokość i wysokość ramki
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# oblicz zarówno początkowe, jak i końcowe (x, y) współrzędne dla ramki granicznej predykcji tekstu
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# dodaj współrzędne ramki granicznej i wynik prawdopodobieństwa do list elementow
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# zwroc tupla z prawdopodobienstwem
	return (rects, confidences)

# konstruktor argumentow
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="sciezka do deketora tekstu EAST")
ap.add_argument("-v", "--video", type=str,
	help="sciezka to pliku wideo")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimalne prawdopodobienstwo wymagane dla regionu")
ap.add_argument("-w", "--width", type=int, default=320,
	help="szerokośc obrazu powinna być wieloktrotnoscia 32")
ap.add_argument("-e", "--height", type=int, default=320,
	help="wysokosc obrazu powinna byc wieloktrotnoscia 32")
args = vars(ap.parse_args())

# inicjalizuj oryginalne wymiary ramki, nowe wymiary ramki i stosunek między nimi
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# #zdefiniuj dwie nazwy warstw wyjściowych dla modelu detektora EAST, którymi jesteśmy zainteresowani - pierwsza to prawdopodobieństwa wyjściowe, a druga może być wykorzystana do uzyskania współrzędnych ramki granicznej tekstu
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] wczytuje detektor tekstu EAST...")
net = cv2.dnn.readNet(args["east"])

# Jeśli nie ma ścieżki do pliku, odpal kamerkę
if not args.get("video", False):
	print("[INFO] uruchamiam strumień wideo...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# lub wczyraj plik wideo
else:
	vs = cv2.VideoCapture(args["video"])

fps = FPS().start()

# loopuj po ramkach streamu wideo
while True:
	# chwyć bieżącą klatkę, a następnie obsłuż ją, jeśli korzystamy z obiektu VideoStream lub VideoCapture
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# wykrywaj text znaleziony na ramce video
	text = pytesseract.image_to_string(frame)
	# drukuj w konsoli
	print('Wykryto tekst: ' + text)

	# sprawdź, czy dotarliśmy do końca strumienia
	if frame is None:
		break

	#zmień rozmiar ramki, zachowując proporcje
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# jeśli nie mamy wymiarow ramek to nadal musimy obliczyć stosunek starych wymiarów ramek do nowych wymiarów
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# zmień rozmiar ramki, tym razem ignorując proporcje
	frame = cv2.resize(frame, (newW, newH))

	# zbuduj bloba z obrazu, a następnie przeslij do przez siec do uzyskania dwóch zestawów warstw wyjściowych
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# zadekoduj prognozy, ignoruj nachodzace ramki - non max suppression
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loopuj po ramkach
	for (startX, startY, endX, endY) in boxes:
		# przeskaluj współrzędne ramki granicznej na podstawie pierwotnych wymiarow ramek
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# rysuj ramke
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
	fps.update()

	# pokaz koncowa ramke
	cv2.imshow("detekcja", orig)
	key = cv2.waitKey(1) & 0xFF

	# wcisnij 'q' aby przerwać
	if key == ord("q"):
		break

# zatrzymaj licznik i pokaż liczbę FPS'ow
fps.stop()
print("[INFO] czas trwania: {:.2f}".format(fps.elapsed()))
print("[INFO] szacowana ilosc FPS: {:.2f}".format(fps.fps()))

# jeśli uzywamy kamerki, uwolnij wskaznik kamerki
if not args.get("video", False):
	vs.stop()

# lub uwolnij wskaznik pliku
else:
	vs.release()

# zniszcz wszystko ;)
cv2.destroyAllWindows()