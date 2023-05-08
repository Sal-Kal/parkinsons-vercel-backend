- Installing Requirements: `pip install -r requirements.txt`

- Starting the server for the first time: `python manage.py makemigrations && python manage.py migrate && python manage.py runserver`

- The server can be stopped using `Ctrl+c` and to start the server again use: `python manage.py runserver`

- API url: `http://localhost:8000/classify/`

- The payload format:
```
Endpoint: /classify

Method: POST

Headers:
  Content-Type: multipart/form-data

Body:
  image: (file) - The image file to be uploaded.
```

- Response:
```
{
  "status": "string",
  "response": {
          "shape": "string",
          "score": "float",
          "prediction": "string",
          "image": "base64 encoded string"
      }
}
```

- The `response.shape` value signifies if the model detected a spiral or a wave.

- The `response.score` value signifies the predicted score, if the value is less than `50` then the person has parkinson's disease.

- The `response.prediction` value signifies if the the person is healthy or has parkinson's disease.

- The `response.image` value is a base64 encoded string which can be decoded to get a png image of the detected spiral or wave within the image.

- The image detection works well only when the drawn spiral or wave is in the center of the image.
