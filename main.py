import base64
import datetime
import os.path
import pickle
import sqlite3
import time

import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, request, Response


class myApp:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        self.db_name = 'face_recognize.db'
        self.table_name = 'FACE_ENCODING'
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        datas = cursor.execute('''SELECT NAME, ENCODING from ''' + self.table_name)
        for (name, base_in) in datas:
            encoding_bytes = base64.b64decode(base_in)
            encoding_face = pickle.loads(encoding_bytes)
            self.known_face_names.append(name)
            self.known_face_encodings.append(encoding_face)
        conn.close()

        self.frame = np.zeros((10, 10), np.uint8)
        self.tolerance = 0.39

        self.app = Flask(__name__)
        self.app.add_url_rule('/', view_func=self.index)
        self.app.add_url_rule('/video_feed', view_func=self.video_feed)
        self.app.add_url_rule('/face_upload', view_func=self.face_upload, methods=['POST'])
        self.app.add_url_rule('/face_recognize', view_func=self.face_recognize, methods=['POST'])
        self.app.add_url_rule('/recognize_result', view_func=self.recognize_result)
        self.app.add_url_rule('/return_index', view_func=self.index, methods=['POST'])

        self.app.run(host="0.0.0.0", port=5002, threaded=True, debug=True)

    def index(self):
        return render_template('index.html', data={'fps': 24}, tolerance=self.tolerance)

    def video_feed(self):
        return Response(self.getImage(), mimetype='multipart/x-mixed-replace;boundary=frame')

    def face_upload(self):
        name = request.values.get('name')
        if name == '':
            return self.index()
        img = request.files['file']
        img.save(name + '.jpg')
        image = face_recognition.load_image_file(name + '.jpg')
        face_encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        encoding_bytes = pickle.dumps(face_encoding)
        base_out = base64.b64encode(encoding_bytes)
        cursor.execute('''INSERT INTO ''' + self.table_name + ''' (NAME,ENCODING) VALUES (?,?)''', (name, base_out))
        conn.commit()
        conn.close()
        return self.index()

    def face_recognize(self):
        self.tolerance = float(request.values.get('tolerance'))
        return render_template('result.html')

    def recognize_result(self):
        return Response(self.getResult(), mimetype='multipart/x-mixed-replace;boundary=frame')

    def getImage(self):
        face_names = []
        face_locations = []
        camera = cv2.VideoCapture(0)
        while True:
            ret, self.frame = camera.read()
            show_img = self.frame.copy()
            read_end = time.time()

            small_frame = cv2.resize(show_img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)

            process_end = time.time()
            cv2.putText(show_img, 'move:%sms' % (round((process_end - read_end) * 1000, 3)), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(show_img, (left << 2, top << 2), (right << 2, bottom << 2), (0, 0, 255), 2)

            show_frame = cv2.imencode('.jpg', show_img)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n')

    def getResult(self):
        show_img = self.frame.copy()
        small_frame = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_names = []
        face_locations = []
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top <<= 1
            right <<= 1
            bottom <<= 1
            left <<= 1
            cv2.rectangle(show_img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(show_img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(show_img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imwrite(os.path.join('result', str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '.jpg'), show_img)
        show_frame = cv2.imencode('.jpg', show_img)[1].tobytes()
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n')


if __name__ == '__main__':
    myApp()
