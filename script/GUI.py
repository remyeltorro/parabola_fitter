#!/usr/bin/env python3

import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFrame, QMessageBox, QTabWidget, QComboBox, QAction, QCheckBox, QTextBrowser, QMenu, QSlider, QFileDialog, QLabel, QPushButton, QLineEdit, QWidget, QDialog, QMainWindow, QGridLayout
from PyQt5.QtGui import QImage, QIcon, QDoubleValidator, QPixmap
from screeninfo import get_monitors
from configparser import ConfigParser
from glob import glob
from natsort import natsorted
from tifffile import imread
import numpy as np
import time
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import gc
import pandas as pd
import shutil
pd.options.mode.chained_assignment = None  # default='warn'
from lmfit import Model,Parameter,Parameters
from lmfit import Minimizer, Parameters, report_fit
from tifffile import imwrite
import cv2

class Window(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("Parabola flattening")
		self.setWindowIcon(QIcon(f"icon/parabola.png"))
		#self.setGeometry(0,0,res_w,res_h)
		self.center_window()

		central_widget = QWidget()
		self.grid = QGridLayout(central_widget)

		self.grid.addWidget(QLabel("Media path:"), 0, 0, 1, 3)

		self.media_path_le = QLineEdit()
		self.media_path_le.setAlignment(Qt.AlignLeft)	
		self.media_path_le.setEnabled(True)
		self.media_path_le.setDragEnabled(True)
		self.media_path_le.setFixedWidth(400)
		self.foldername = os.getcwd()
		self.media_path_le.setText(self.foldername)
		self.grid.addWidget(self.media_path_le, 1, 0, 1, 3)

		self.browse_button = QPushButton("Browse...")
		self.browse_button.clicked.connect(self.browse_experiment_folder)
		self.grid.addWidget(self.browse_button, 1, 4, 1, 1)

		self.grid.addWidget(QLabel("Slicing interval: "), 2,0,1,1)

		self.interval_le = QLineEdit()
		self.interval_le.setAlignment(Qt.AlignRight)	
		self.interval_le.setEnabled(True)
		self.interval_le.setText("1")
		#self.interval_le.setFixedWidth(400)
		self.grid.addWidget(self.interval_le, 2, 1, 1, 2)


		self.validate_button = QPushButton("Submit")
		self.validate_button.clicked.connect(self.load_stack)
		self.grid.addWidget(self.validate_button, 4, 1, 1, 1, alignment = Qt.AlignCenter)
		
		self.setCentralWidget(central_widget)
		self.show()

	def center_window(self):
		frameGm = self.frameGeometry()
		screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
		centerPoint = QApplication.desktop().screenGeometry(screen).center()
		frameGm.moveCenter(centerPoint)
		self.move(frameGm.topLeft())

	def browse_experiment_folder(self):
		foldername = QFileDialog.getOpenFileName(self, 'Select file')
		self.media_path_le.setText(foldername[0])

	def load_stack(self):
		self.media_path = self.media_path_le.text()
		self.media_name = os.path.split(self.media_path)[-1]
		self.file_extension = self.media_name.split(".")[-1]

		try:
			self.interval = int(self.interval_le.text())
		except:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("Please set an integer value for interval.")
			msgBox.setWindowTitle("Error")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return(None)				


		if self.file_extension=="tif":
			print("TIF file detected. Performing flattening on static image.")
			self.capture = imread(self.media_path)
			if len(self.capture.shape)==2:
				self.static_tif_mode()
			elif len(self.capture.shape)==3:
				self.multi_position_tif_mode()

		elif self.file_extension=="avi":
			print(f"AVI file detected. Performing flattening every {self.interval} frames.")
			self.avi_mode()

		else:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("Please set a valid input file (.tif or .avi).")
			msgBox.setWindowTitle("Error")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return(None)

	def fit_parabola(self, image):
		data = np.empty(image.shape)
		x = np.arange(0,image.shape[1])
		y = np.arange(0,image.shape[0])
		xx,yy = np.meshgrid(x,y)

		def parabola(x,y,x0,y0,a,b):
			return((x-x0)**2/a**2 + (y-y0)**2/b**2)

		params = Parameters()
		params.add('x0', value=1)
		params.add('y0', value=6)
		params.add('a', value=3)
		params.add('b', value=1)

		model = Model(parabola,independent_vars=['x','y'])

		result = model.fit(image, x=xx, y=yy, params=params)
		print(report_fit(result))

		x0 = result.params['x0'].value
		y0 = result.params['y0'].value
		a = result.params["a"].value
		b = result.params["b"].value

		para = parabola(xx,yy,x0,y0,a,b)
		
		return(para)

	def static_tif_mode(self):

		print("Fitting the parabola on the image...")
		para = self.fit_parabola(self.capture)

		output_dir = os.path.split(self.media_path)[0]+"/"+os.path.split(self.media_path)[-1][:-4]
		name = os.path.split(self.media_path)[-1][:-4]

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		plt.imshow(self.capture,cmap="gray")
		plt.imshow(para,alpha=0.3)
		plt.colorbar()
		plt.pause(3)
		plt.close()

		correction = self.capture - para + np.amax(para)
		imwrite(output_dir+f"/{name}_flat.tif", np.array(correction,dtype=np.int32))
		print(f"Flattened image saved in "+ output_dir+f"/{name}_flat.tif")
		plt.title("Diagonal slice")
		plt.plot([correction[i,i] for i in range(np.amin(np.shape(correction)))],label="Flattened")
		plt.plot([self.capture[i,i] for i in range(np.amin(np.shape(self.capture)))],label="Original")
		plt.legend()
		plt.pause(2)
		plt.close()

	def multi_position_tif_mode(self):

		output_dir = os.path.split(self.media_path)[0]+"/"+os.path.split(self.media_path)[-1][:-4]
		name = os.path.split(self.media_path)[-1][:-4]

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		for j,pos in enumerate(self.capture):

			para = self.fit_parabola(pos)

			plt.imshow(pos,cmap="gray")
			plt.imshow(para,alpha=0.3)
			plt.colorbar()
			plt.pause(3)
			plt.close()

			correction = pos - para + np.amax(para)
			imwrite(output_dir+f"/{name}_flat_{j}.tif", np.array(correction/np.mean(correction)*1000,dtype=np.int32))
			print(f"Flattened image saved in "+ output_dir+f"/{name}_flat.tif")
			plt.title("Diagonal slice")
			plt.plot([correction[i,i] for i in range(np.amin(np.shape(correction)))],label="Flattened")
			plt.plot([pos[i,i] for i in range(np.amin(np.shape(pos)))],label="Original")
			plt.legend()
			plt.pause(2)
			plt.close()			


	def avi_mode(self):

		capture = cv2.VideoCapture(self.media_path)
		nbr_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
		fps = capture.get(cv2.CAP_PROP_FPS)
		print(f"Number of frames: {nbr_frames}")
		print(f"FPS before flattening: {fps}")
		print(f"FPS after flattening: {float(fps/self.interval)}")

		output_dir = os.path.split(self.media_path)[0]+"/"+os.path.split(self.media_path)[-1][:-4]
		name = os.path.split(self.media_path)[-1][:-4]

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		success,image = capture.read()
		count = 0
		stack = []

		while success:
			if (count % self.interval)==0:
				print(f"Frame: {count}")
				stack.append(image[:,:,0]) #take 1 channel from RGB
			success,image = capture.read()
			count += 1

		print("Generating median stack...")
		median = np.median(stack,axis=0)
		print("Fitting the parabola on the median stack...")
		para = self.fit_parabola(median)

		plt.imshow(median,cmap="gray")
		plt.imshow(para,alpha=0.3)
		plt.colorbar()
		plt.pause(3)
		plt.close()

		for k in tqdm(range(len(stack))):
			frame = stack[k]
			correction = frame - para + np.amax(para)
			imwrite(output_dir+f"/{name}_flat_{int(k*self.interval)}.tif", np.array(correction,dtype=np.int16))

		print(f"Flattened image saved in "+ output_dir+f"/{name}_flat_i.tif")
		plt.title("Diagonal slice")
		plt.plot([correction[i,i] for i in range(np.amin(np.shape(correction)))],label="Flattened")
		plt.plot([median[i,i] for i in range(np.amin(np.shape(median)))],label="Median stack")
		plt.legend()
		plt.pause(2)
		plt.close()

App = QApplication(sys.argv)
App.setStyle("Fusion")
window = Window()
sys.exit(App.exec())