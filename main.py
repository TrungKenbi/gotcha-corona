import asyncio
import random
import string

import cv2
import numpy as np
import websockets
import base64
import json
import os
import tensorflow.keras
from PIL import Image, ImageOps

CORONA_TEMPLATE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/corona_template.png'
CORONA_SCALE_RATIO = 0.5

corona_template_image = cv2.imread(CORONA_TEMPLATE_PATH, 0)
corona_template_image = cv2.resize(corona_template_image, None, fx=CORONA_SCALE_RATIO, fy=CORONA_SCALE_RATIO)

scan_folder = 'scan'
scan_images = []
for filename in os.listdir(scan_folder):
    img = cv2.imread(os.path.join(scan_folder, filename), 0)
    img = cv2.resize(img, None, fx=CORONA_SCALE_RATIO, fy=CORONA_SCALE_RATIO)
    # img = cv2.cvtColor(wave_image, cv2.COLOR_BGRA2GRAY)
    if img is not None:
        scan_images.append(img)

# Classification
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')

def catch_corona(wave_image, threshold=0.1):
    wave_image_gray = cv2.cvtColor(wave_image, cv2.COLOR_BGRA2GRAY)

    result = []
    for scan in scan_images:
        res = cv2.matchTemplate(wave_image_gray, scan, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < threshold:
            continue

        width, height = scan.shape[::-1]
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)

        corona_img_original = wave_image[max_loc[1]:top_left[1] + height, max_loc[0]:top_left[0] + width]

        size = (224, 224)
        corona_img = ImageOps.fit(Image.fromarray(corona_img_original), size, Image.ANTIALIAS)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_array = np.asarray(corona_img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        if prediction[0][0] >= 0.003:
            continue

        if prediction[0][1] < 0.999999:
            continue

        result.append([top_left, bottom_right])

    return result


def base64_to_image(base64_data):
    encoded_data = base64_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    return img


async def play_game(websocket, path):
    print('Corona Killer is ready to play!')
    catchings = []
    last_round_id = ''
    wave_count = 0

    while True:

        ### receive a socket message (wave)
        try:
            data = await websocket.recv()
        except Exception as e:
            print('Error: ' + e)
            break

        json_data = json.loads(data)

        ### check if starting a new round
        if json_data["roundId"] != last_round_id:
            print(f'> Catching corona for round {json_data["roundId"]}...')
            last_round_id = json_data["roundId"]

        ### catch corona in a wave image
        wave_image = base64_to_image(json_data['base64Image'])
        results = catch_corona(wave_image)

        ### save result image file for debugging purpose
        for result in results:
            cv2.rectangle(wave_image, result[0], result[1], (0, 0, 255), 2)

        waves_dir = f'waves/{last_round_id}/'
        if not os.path.exists(waves_dir):
            os.makedirs(waves_dir)

        # cv2.imwrite(os.path.join(waves_dir, f'{json_data["waveId"]}.jpg'), wave_image)

        print(f'>>> Wave #{wave_count:03d}: {json_data["waveId"]}')
        wave_count = wave_count + 1

        ### store catching positions in the list
        catchings.append({
            "positions": [
                {"x": (result[0][0] + result[1][0]) / 2, "y": (result[0][1] + result[1][1]) / 2} for result in results
            ],
            "waveId": json_data["waveId"]
        })

        ### send result to websocket if it is the last wave
        if json_data["isLastWave"]:
            round_id = json_data["roundId"]
            print(f'> Submitting result for round {round_id}...')

            json_result = {
                "roundId": round_id,
                "catchings": catchings,
            }

            await websocket.send(json.dumps(json_result))
            print('> Submitted.')

            catchings = []
            wave_count = 0


start_server = websockets.serve(play_game, "localhost", 8765, max_size=100000000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
