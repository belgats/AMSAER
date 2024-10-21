# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	'AMSAER': '/home/slasher/AMSAER-Baseline/dataset-process',
}
PATH_TO_RAW_AUDIO = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'], 'audio'),
}
PATH_FROM_RAW_FACE = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'],   'video'),
}
PATH_TO_RAW_FACE = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'], 'features'),
}
PATH_TO_LABEL = {
	'AMSAER': os.path.join(DATA_DIR['AMSAER'], 'label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = '/home/slasher/AMSAER-Baseline/tools/'
PATH_TO_OPENSMILE = '/home/slasher/opensmile'
PATH_TO_FFMPEG = '/home/slasher/anaconda3/bin/ffmpeg'
PATH_TO_NOISE = '/share/home/ /tools/musan/audio-select'
PATH_TO_OPENFACE = ''

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (openface-win) ##############
DATA_DIR_Win = {
	'AMSAER': '/home/slasher/AMSAER-Baseline/dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'AMSAER':   os.path.join(DATA_DIR_Win['AMSAER'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'AMSAER':   os.path.join(DATA_DIR_Win['AMSAER'],   'features'),
}

PATH_TO_OPENFACE_Win = "/home/slasher/AMSAER-Baseline/tools/openface/build/bin/"