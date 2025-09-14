from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd



   
class Calibrate(QtWidgets.QDialog):
    def __init__(self,args):
        
        super(Calibrate, self).__init__()
        self.ui=uic.loadUi('calibrate.ui', self)
        self._args=args

        self.onCancel.clicked.connect(self.onCancelEvent)
        self.onSave.clicked.connect(self.onSaveEvent)
        self.selectFile.clicked.connect(self.onSelectOutputFile)
        self.runCalibration.setChecked(args.calibrate)
        self.checkSync.setChecked(args.synchronize_index)
        self.checkSyncAxis.setChecked(args.synchronize_axis)
        self.checkLSQ.setChecked(args.uselsq)
        self.checkOpt.setChecked(args.useopt)
        self.CalibrationSamplePeriodicityInMinutes.setValue(args.CalibrationSamplePeriodicityInMinutes)
        self.CalibrationPeriodicityInHours.setValue(args.CalibrationPeriodicityInHours)
        self.MinimumFlowToSample.setValue(args.MinimumFlowToSample)
        self.NumberOfSamplesForCalibration.setValue(args.NumberOfSamplesForCalibration)
        self.CalibrationDelayAfterRestartInHours.setValue(args.CalibrationDelayAfterRestartInHours)
        self.CalibrationOffsetSamplesFromCurrentTime.setValue(args.CalibrationOffsetSamplesFromCurrentTime)

        self.show()

    def saveFileNameDialog(self,ext="Text Files (*.txt);;All Files (*)"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save file","",ext, options=options)
        return fileName

 
    def onSelectOutputFile(self,ext="All Files (*)"):
        file=self.saveFileNameDialog("CSV Files (*.csv);;All Files (*)")
        if not file is None and file!="":
            self.outputFile.setText(file)
 
    def onSaveEvent(self):
        self._args.calibrate=self.runCalibration.isChecked()
        self._args.synchronize_axis=self.checkSyncAxis.isChecked()
        self._args.synchronize_index=self.checkSync.isChecked()
        self._args.uselsq=self.checkLSQ.isChecked()
        self._args.useopt=self.checkOpt.isChecked()
        self._args.CalibrationSamplePeriodicityInMinutes=self.CalibrationSamplePeriodicityInMinutes.value()
        self._args.CalibrationPeriodicityInHours=self.CalibrationPeriodicityInHours.value()
        self._args.MinimumFlowToSample=self.MinimumFlowToSample.value()
        self._args.NumberOfSamplesForCalibration=self.NumberOfSamplesForCalibration.value()
        self._args.CalibrationDelayAfterRestartInHours=self.CalibrationDelayAfterRestartInHours.value()
        self._args.CalibrationOffsetSamplesFromCurrentTime=self.CalibrationOffsetSamplesFromCurrentTime.value()
        self._args.calibration_output=None if self.outputFile.text()=="" else self.outputFile.text()
        self.close()

    def onCancelEvent(self):
        self.close()


