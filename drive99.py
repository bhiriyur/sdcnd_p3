"""
Based on:
https://github.com/dolaameng/Udacity-SDC_Behavior-Cloning/tree/master/sdc
"""
import base64
from flask import Flask, render_template
from io import BytesIO
from r99 import process_image, model
import eventlet
import eventlet.wsgi
import numpy as np
import socketio

sio = socketio.Server()
app = Flask(__name__)
target_speed = 22
shape = (100, 100, 3)
model = model(True, shape)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current image from the center camera of the car
    img_str = data["image"]
    speed = float(data["speed"])

    # Set the throttle.
    throttle = 1.2 - (speed / target_speed)

    # read and process image
    image_bytes = BytesIO(base64.b64decode(img_str))
    image, _ = process_image(image_bytes, None, False)

    # make prediction on steering
    sa = model.predict(np.array([image]))[0][0]

    print(sa, throttle)
    send_control(sa, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)