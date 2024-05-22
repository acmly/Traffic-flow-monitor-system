import base64
import cv2
import os
import json
import string
import random
import jwt

import numpy as np
import supervision as sv

import time

import datetime
from typing import Any
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, request, abort, send_from_directory, jsonify, session
from pprint import pprint
from apscheduler.schedulers.background import BackgroundScheduler

from ultralytics import YOLO

from utils.flask_utils import *

load_dotenv(override=True, dotenv_path='config/end-back.env')

HOST_NAME = os.environ['HOST_NAME']
PORT = int(os.environ['PORT'])
TOLERANT_TIME_ERROR = int(os.environ['TOLERANT_TIME_ERROR'])

current_dir = os.getcwd()
BEFORE_IMG_PATH = os.path.join(current_dir, 'static', os.environ['BEFORE_IMG_PATH'])
AFTER_IMG_PATH = os.path.join(current_dir, 'static', os.environ['AFTER_IMG_PATH'])

MYSQL_HOST = os.environ['MYSQL_HOST']        
MYSQL_PORT = os.environ['MYSQL_PORT']           
MYSQL_user = os.environ['MYSQL_user']            
MYSQL_password = os.environ['MYSQL_password']    
MYSQL_db = os.environ['MYSQL_db']               
MYSQL_charset = os.environ['MYSQL_charset']     

db = SQLManager(host=MYSQL_HOST, port=eval(MYSQL_PORT), user=MYSQL_user,
				passwd=MYSQL_password, db=MYSQL_db, charset=MYSQL_charset)
# result = db.get_one("SELECT * FROM user WHERE username=%s", ('dzp'))
# pprint(result)
# pprint(result['age'])

# Load a model
model = YOLO("./models/car.pt")  # load a pretrained model (recommended for training)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

app = Flask(__name__, static_folder='static')
scheduler = BackgroundScheduler()
def scheduled_function():
    print('Scheduled task launch！')
    select_sql = "SELECT id, threshold, url, is_alarm, mode, location " \
          "FROM monitor WHERE is_alarm='on'"
    monitor_list = db.get_list(select_sql)
    for item in monitor_list:
        pid = int(item['id'])
        threshold = int(item['threshold'])
        mode = item['mode']
        location = item['location']
        source = item['url']

        if not check_stream_availability(source):
            print(f'This stream pull failed:{source}')
            return False

        if mode == "Fast mode":
            iter_model = iter(
                model.track(source=source, show=False, stream=True, iou=0.3, conf=0.3))
        elif mode == "Precision Mode":
            iter_model = iter(
                model.track(source=source, show=False, stream=True, iou=0.7, conf=0.7))
        for i in range(2):
            result = next(iter_model) 
            detections = sv.Detections.from_yolov8(result)
            if result.boxes.id is None:
                continue
            if len(detections) > threshold:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                res_url = save_res_img(result.orig_img, detections, f'alarm.jpg')
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                alarm_description = f'Traffic flow:{len(detections)}'
                insert_sql = "INSERT INTO alarm (location, description, threshold, photo, pid, create_time, remark) " \
                             "VALUES (%s, %s, %s, %s, %s, %s, %s)"

                db.modify(insert_sql, (location, alarm_description, threshold, res_url, pid, current_time, 'NULL'))
                print('Alarm logged！')

scheduler.add_job(scheduled_function, 'interval', seconds=20 * 1)
scheduler.start()

app.config['SECRET_KEY'] = 'my-secret-key' 
app.config['PERMANENT_SESSION_LIFETIME'] = 15 * 60 
whitelist = ['/', '/login', '/photo', '/recognize']
@app.before_request
def interceptor():
    if request.path.startswith('/static/'): 
        return
    if request.path in whitelist: 
        return
    if not session.get('username'): 
        return wrap_unauthorized_return_value('Unauthorized') 


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response

@app.route("/")
def start_server():
    return "Welcome to the Traffic Road Analysis System! Successful startup of the backend！(*^▽^*)"

@app.route('/login', methods=["POST"])
def login():
    try:
        data = request.json 
        username = data.get('username').strip()
        password = data.get('password').strip()
        user_info = db.get_one("SELECT * FROM user WHERE username=%s", (username))
        if user_info and user_info['password'] == password:
            session['username'] = username 
            return wrap_ok_return_value({'id':user_info['id'],
                                         'avatar':user_info['avatar'],
                                         'username':user_info['username']})
        return wrap_error_return_value('Wrong username or password！') 
    except:
        return wrap_error_return_value('The system is busy, please try again later！')

@app.route('/logOut', methods=["get"])
def log_out():
    session.clear()
    return wrap_ok_return_value('Account has been logged out！')

@app.route('/submitMonitorForm', methods=["POST"])
def submit_monitor_form():
    try:
        data = request.json 
        threshold = int(data.get('threshold'))
        person = data.get('person')
        video = data.get('video')
        url = data.get('url')
        if(data.get('is_alarm')):
            is_alarm = 'on'
        else:
            is_alarm = 'off'
        mode = data.get('mode')
        location = data.get('location')
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remark = data.get('remark')
        insert_sql = "INSERT INTO monitor " \
              "(threshold, person, video, url, is_alarm, mode, location, create_time, create_by, remark) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (threshold, person, video, url, is_alarm, mode, location, create_time, "", remark)
        db.modify(insert_sql, values)
        return wrap_ok_return_value('Configuration submitted successfully！')
    except Exception as e:
        pprint(e)
        return wrap_error_return_value('The system is busy, please try again later！')


@app.route('/updateMonitorForm', methods=["POST"])
def update_monitor_form():
    try:
        data = request.json 
        id = data.get('id')
        threshold = int(data.get('threshold'))
        person = data.get('person')
        video = data.get('video')
        url = data.get('url')
        if(data.get('is_alarm')):
            is_alarm = 'on'
        else:
            is_alarm = 'off'
        mode = data.get('mode')
        location = data.get('location')
        remark = data.get('remark')

        update_sql = "UPDATE monitor SET " \
                     "threshold = %s, person = %s, video = %s, url = %s, " \
                     "is_alarm = %s, mode = %s, location = %s, remark = %s " \
                     "WHERE id = %s"
        values = (threshold, person, video, url, is_alarm, mode, location, remark, id)
        db.modify(update_sql, values)

        return wrap_ok_return_value('Configuration update successful！')

    except Exception as e:
        return wrap_error_return_value(str(e))

@app.route('/usersList/<int:page>', methods=['GET'])
def get_user_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"select id, username, avatar, email, grade from user limit {page_from}, {page_to}"
    user_list = db.get_list(select_sql)
    return wrap_ok_return_value(user_list)

@app.route('/monitorList/<int:page>', methods=['GET'])
def get_monitor_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"SELECT id, threshold, person, video, url, is_alarm, mode, " \
          f"location, create_time, create_by, remark FROM monitor" \
          f" limit {page_from}, {page_to}"
    monitor_list = db.get_list(select_sql)
    for item in monitor_list:
        item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')
    return wrap_ok_return_value(monitor_list)

@app.route('/alarmList/<int:page>', methods=['GET'])
def get_alarm_list(page):
    page_from = int((page - 1) * 10)
    page_to = int(page)*10
    select_sql = f"SELECT id, location, description, threshold, photo, pid, create_time, remark " \
                 f"FROM alarm LIMIT {page_from}, {page_to}"
    alarm_list = db.get_list(select_sql)

    for item in alarm_list:
        item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')

    return wrap_ok_return_value(alarm_list)

@app.route("/photo", methods=["POST"])
def recognize_base64():
    photo_data = request.form.get('photo')
    photo_data = photo_data.replace('data:image/png;base64,', '') 

    image_data = base64.b64decode(photo_data)

    before_img_path = save_img_base64(image_data, path=BEFORE_IMG_PATH)

    name = f"{''.join(random.choice(string.ascii_lowercase) for i in range(5))}.png"

    return yolo_res(before_img_path=before_img_path, name=name)

@app.route("/recognize", methods=["POST"])
def recognize_photo():
    photo = request.files['file']
    name = photo.filename
    img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    before_img_path = save_img(name,img,BEFORE_IMG_PATH)

    return yolo_res(before_img_path=before_img_path, name=name)

def yolo_res(before_img_path, name):

    try:
        img = Image.open(before_img_path)
        iter_model = iter(
            model.track(source=img, show=False))
        result = next(iter_model) 

        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is None:
            return wrap_ok_return_value('No target object in the photo！')
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        res_img = result.orig_img
        res_url = save_res_img(res_img, detections)

        labels = [
            f"OBJECT-ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f} x:{x} y:{y}"
            for x, y, confidence, class_id, tracker_id in detections
        ]

        return wrap_ok_return_value({
            'labels': labels,
            'after_img_path': res_url
        })
    except Exception as e:
        pprint(str(e))
        return wrap_error_return_value('The server is busy, please try again later！')

def save_res_img(res_img, detections, name = 'default.jpg'):

    labels = [
        f"ID: {tracker_id}"
        for x, y, confidence, class_id, tracker_id in detections
    ]

    img_box = box_annotator.annotate(scene=res_img, detections=detections, labels=labels)

    rgb_frame = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)

    numpy_frame = np.array(rgb_frame)

    after_img_path = save_img(name, numpy_frame, AFTER_IMG_PATH)

    return after_img_path.replace(current_dir, "http://127.0.0.1:5500/").replace('\\', '/')

if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)
