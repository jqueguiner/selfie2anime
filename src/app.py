import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file
import traceback

from app_utils import blur
from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin
from app_utils import unzip
from app_utils import unrar
from app_utils import resize_img
from app_utils import square_center_crop
from app_utils import square_center_crop
from app_utils import image_crop


from UGATIT import UGATIT
import argparse
from utils import *
import matplotlib.image as mpimg
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
import face_recognition
from PIL import *


try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class FaceCropper(object):
    CASCADE_PATH = "/src/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, output_path, padding, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        height, width = img.shape[:2]

        (x, y, w, h) = faces[0]
        r = min((max(w, h) / 2)*(100+padding)/100, width, height)
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (256, 256))
        cv2.imwrite(output_path, lastimg)


def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


@app.route("/process", methods=["POST"])
def process():

    input_path = generate_random_filename(upload_directory,"jpg")
    output_path = generate_random_filename(result_directory,"jpg")

    try:
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
            
        else:
            url = request.json["url"]
            download(url, input_path)

        try:
            detecter = FaceCropper()
            detecter.generate(input_path, input_path, 20, False)
            #
            print("face_crop")
        except:
            im = Image.open(input_path)
            
            width, height = im.size

            print(im.size)
            center_x = width/2
            center_y = height/2

            box_size = min(width, height)

            print(box_size)
            left = center_x - box_size/2
            top = center_y - box_size/2
            width = box_size
            height = box_size
          
            box = (left, top, left + width, top + height)

            area = im.crop(box)

            area = im.convert("RGB")

            area.save(input_path, "JPEG", quality=80, optimize=True, progressive=True)
            print("image_crop")



        gan.test_endpoint(input_path, output_path)


        callback = send_file(output_path, mimetype='image/jpeg')

        return callback, 200


    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory
    global model_directory
    global args
    global gan
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    result_directory = '/src/results/'
    create_directory(result_directory)

    upload_directory = '/src/UGATIT/dataset/selfie2anime/testA/'
    create_directory(upload_directory)

    create_directory('/src/UGATIT/dataset/selfie2anime/testB/')
    create_directory('/src/UGATIT/dataset/selfie2anime/trainA/')
    create_directory('/src/UGATIT/dataset/selfie2anime/trainB/')

    model_directory = '/src/checkpoint/'
    create_directory(model_directory)


    url_prefix = 'http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra.cloud.ovh.net/image/'

    model_file_rar = 'UGATIT_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing.rar'

    haarcascade_file = 'haarcascade_frontalface_default.xml'

    get_model_bin(url_prefix + "ugatit/selfie2anime/" + model_file_rar , os.path.join('/src', model_file_rar))
    unrar(model_file_rar, model_directory)

    get_model_bin(url_prefix + "haarcascade/" + haarcascade_file,  os.path.join('/src', haarcascade_file))
    
    args = parse_args()
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1))

    gan = UGATIT(sess, args)
    gan.build_model()

    gan.test_endpoint_init()


    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=False)

