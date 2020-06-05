# uzycie
# python3 text_detection.py --image zdjecia/costam.jpg --east frozen_east_text_detection.pb

# importujemy paczki
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# kontruktor argumentow
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="sciezka do pliku")
ap.add_argument("-east", "--east", type=str,
	help="sciezka do detektora tekstu EAST")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimalne prawdopodobienstwo regionu obrazu")
ap.add_argument("-w", "--width", type=int, default=320,
	help="szerokośc obrazu powinna być wieloktrotnoscia 32")
ap.add_argument("-e", "--height", type=int, default=320,
	help="wysokosc obrazu powinna byc wieloktrotnoscia 32")
args = vars(ap.parse_args())

# wczytaj obraz i jego rozdzielczosc
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# ustaw nową szerokość i wysokość, a następnie określ współczynnik zmiany dla szerokości i wysokości
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# zmień rozmiar obrazu i pobierz nowe wymiary obrazu
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

#zdefiniuj dwie nazwy warstw wyjściowych dla modelu detektora EAST, którymi jesteśmy zainteresowani - pierwsza to prawdopodobieństwa wyjściowe, a druga może być wykorzystana do uzyskania współrzędnych ramki granicznej tekstu
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# wczytaj pretrenowant detektor EAST
print("[INFO] wczytuje detektor EAST...")
net = cv2.dnn.readNet(args["east"])

# zbuduj bloba z obrazu, a następnie przeslij do przez siec do uzyskania dwóch zestawów warstw wyjściowych
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# pokaż informacje o taktowaniu w przewidywaniu tekstu
print("[INFO] detekcja tekstu zajela {:.6f} sekund".format(end - start))

# pobierz liczbę wierszy i kolumn z objętości wyników, a następnie zainicjalizuj ramki i odpowiadające im wyniki prawdopodobienstwa
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loopuj po tablicy
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

		# oblicz współczynnik przesunięcia, ponieważ nasze wynikowe mapy obiektów będą czterokrotnie mniejsze niż obraz wejściowy
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# użyj geometrii, aby wyliczyć szerokość i wysokość ramki
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# oblicz zarówno początkowe, jak i końcowe (x, y) współrzędne dla ramki granicznej przewidywania tekstu
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# dodaj współrzędne ramki  i wynik prawdopodobieństwa do odpowiednich pozycji z listy
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# ignoruj nachodzace ramki - non max suppression
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loopoj po ramkach
for (startX, startY, endX, endY) in boxes:
	# przeskaluj współrzędne ramki granicznej na podstawie odpowiedniego wskaźnika
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# narysuj ramki na obrazach
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# pokaz obraz
cv2.imshow("wykrycie tekstu", orig)
cv2.waitKey(0)