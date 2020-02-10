#!/usr/bin/env python
#
# Project: Object Detection using Haar feature-based in live Video Streaming with Flask
# Author: Arad Fumm
# Date: 2020/02/10
# Websites:[
#   http://www.chioka.in/
#   https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
#   http://diymaker.me
# ]
# Description:
#  live Object Detection in video stream 
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. in camera.py edit where is cv2 Haar located
# 4. Navigate the browser to the local webpage.

from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)