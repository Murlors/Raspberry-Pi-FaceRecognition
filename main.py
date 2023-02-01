import base64
import datetime
import os.path
import pickle
import sqlite3
import time

import cv2
import face_recognition
from flask import Flask, render_template, request, Response

DATABASE = 'face_recognize.db'
TABLE_NAME = 'FACE_ENCODING'
app = Flask(__name__, template_folder='templates')
known_face_names = []
known_face_encodings = []
# 公差
tolerance = 0.39
frame = None


@app.route("/", "/return_index")
def index():
    return render_template('index.html', data={'fps': 24}, tolerance=tolerance)


@app.route("/video_feed")
def video_feed():
    return Response(get_image(), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route("/face_upload", methods=['POST'])
def face_upload():
    name = request.values.get('name')
    if name == '':
        return index()
    img = request.files['file']
    img.save(f'{name}.jpg')

    image = face_recognition.load_image_file(f'{name}.jpg')
    # 在上传的图片中选第一个识别到的脸
    face_encoding = face_recognition.face_encodings(image)[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    # 将人脸数据放入数据库
    with sqlite3.connect(DATABASE).cursor() as cursor:
        # pickle把脸部数据对象编码成二进制数据
        encoding_bytes = pickle.dumps(face_encoding)
        # 将二进制数据进行base64编码
        base_out = base64.b64encode(encoding_bytes)
        cursor.execute('INSERT INTO (?) (NAME,ENCODING) VALUES (?,?)''', (TABLE_NAME, name, base_out))
        cursor.commit()

    return index()


@app.route("/face_recognize", methods=['POST'])
def face_recognize():
    tolerance = float(request.values.get('tolerance'))
    return render_template('result.html')


@app.route("/recognize_result")
def recognize_result():
    return Response(get_result(), mimetype='multipart/x-mixed-replace;boundary=frame')


def get_image():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        show_img = frame.copy()
        read_end = time.time()
        # 减小图片大小加速处理
        small_frame = cv2.resize(show_img, (0, 0), fx=0.25, fy=0.25)
        # 将图片由BGR转为RGB
        rgb_small_frame = small_frame[:, :, ::-1]

        # 寻找人脸位置
        face_locations = face_recognition.face_locations(rgb_small_frame)
        process_end = time.time()
        cv2.putText(show_img, f'move:{round((process_end - read_end) * 1000, 3)}ms', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 人脸周围画框
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(show_img, (left << 2, top << 2), (right << 2, bottom << 2), (0, 0, 255), 2)

        show_frame = cv2.imencode('.jpg', show_img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n')


def get_result():
    # 减小图片大小加速处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # 将图片由BGR转为RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    face_names = []
    # 寻找人脸位置
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # 对找到的人脸进行编码
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    # 与数据库中的人脸进行匹配
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    # 在识别到的人脸上加上信息
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top << 1, right << 1, bottom << 1, left << 1
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('result', f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.jpg'), frame)

    show_frame = cv2.imencode('.jpg', frame)[1].tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + show_frame + b'\r\n')


if __name__ == '__main__':
    # 读取数据库
    with sqlite3.connect(DATABASE).cursor() as cursor:
        datas = cursor.execute('SELECT NAME, ENCODING from (?) ', TABLE_NAME)
        for name, base_in in datas:
            encoding_bytes = base64.b64decode(base_in)
            encoding_face = pickle.loads(encoding_bytes)
            known_face_names.append(name)
            known_face_encodings.append(encoding_face)

    app.run(host="0.0.0.0", port=5002, threaded=True, debug=True)
