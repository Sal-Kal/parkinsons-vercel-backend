from rest_framework.decorators import api_view
from django.http import JsonResponse
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

# Loading the ml model
spiral_model = load_model('./ml-model/spiral.h5')
wave_model = load_model('./ml-model/wave.h5')

# Dimensions for Resizing
spiral_height = 256
spiral_width = 256

wave_height = 256
wave_width = 512

@api_view(['POST'])
def classify(request):
    try:
        response = {}
        image = request.FILES['image'].read()

        if "bg" in request.data:
            bg = request.data["bg"]
            bg = bg.split()
            bg = [int(x) for x in bg]
        else:
            bg = [255, 255, 255]

        if "fg" in request.data:
            fg = request.data["fg"]
            fg = fg.split()
            fg = [int(x) for x in fg]
        else:
            fg = [0, 0, 0]

        nparr = np.frombuffer(image, np.uint8)
        drawing = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # Calculating the center of the image
        center = (drawing.shape[1]/2, drawing.shape[0]/2)
        height, width, _ = drawing.shape

        # Calculating the 80% of Image area
        area = height * width
        threshold_area = 0.8 * area

        # Finding Contours
        gray = cv.cvtColor(drawing, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Sorting the Contours based on area of the Contour
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # Selecting the top 4 contours with most area
        rects = contours[0:4]

        # Finding the Contour which is closest to the middle of the image
        min_distance = float('inf')
        closest = None

        for contour in rects:
            # Find the bounding rectangle for the contour
            x, y, w, h = cv.boundingRect(contour)
            rect_area = w * h
            # Skip contour if area of the contour is more than or equal to 80% of the image
            if rect_area >= threshold_area:
                continue

            # Find the center of the bounding rectangle
            rect_center = (x + w/2, y + h/2)

            # Calculate the distance between the center of the rectangle and the center of the image
            distance = ((rect_center[0] - center[0])**2 + (rect_center[1] - center[1])**2)**0.5

            # Check if the current rectangle is closer to the center of the image than the previous closest rectangle
            if distance < min_distance:
                min_distance = distance
                closest = contour

        x, y, w, h = cv.boundingRect(closest)

        crop_img = drawing[y:y+h, x:x+w]

        aspect_ratio = float(w) / h if h != 0 else float('inf')

        # Checking if the image is more likely to be a spiral or a wave
        if aspect_ratio >= 1.7:
            crop_img = cv.resize(crop_img, (wave_width, wave_height))
            print(aspect_ratio)
            is_spiral = 0
        else:
            crop_img = cv.resize(crop_img, (spiral_width, spiral_height))
            print(aspect_ratio)
            is_spiral = 1

        processed = tf.image.resize(cv.cvtColor(crop_img, cv.COLOR_BGR2RGB), (256, 256))

        if (is_spiral == 1):
            # Spiral
            prediction = spiral_model.predict(np.expand_dims(processed/255,0))
            response["shape"] = "Spiral"
        else:
            # Wave
            prediction = wave_model.predict(np.expand_dims(processed/255,0))
            response["shape"] = "Wave"

        score = float(prediction[0][0])
        score = score * 100
        score = 100 - score

        response["score"] = "{:.1f}".format(score)

        if prediction > 0.5:
            response["prediction"] = "Parkinson"
        else:
            response["prediction"] = "Healthy"

        crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

        _, thresh_img = cv.threshold(crop_img, 150, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        img_color = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype=np.uint8)
        img_color[thresh_img == 255] = (bg[2], bg[1], bg[0])
        img_color[thresh_img == 0] = (fg[2], fg[1], fg[0])

        _, buffer = cv.imencode('.png', img_color)
        encoded = base64.b64encode(buffer).decode('utf-8')

        response["image"] = encoded

        return JsonResponse({
            "status" : "success",
            "response" : response
        })

    except Exception as e:
        return JsonResponse({
            "status" : "error",
            "response" : str(e)
        })
