from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import flow_analyzer as fr
import os
import matplotlib
import pandas as pd
import datetime
import calibrategui
matplotlib.use('Qt5Agg')



class Args:
    input = ""
    output = None
    table = None
    frm = None
    to = None
    display_charts_1 = False
    display_charts_2 = False
    display_charts_3 = 0
    display_charts_4 = False
    display_charts_5 = False
    display_grid = False
    abs_value = False
    bias1 = 0.0
    bias2 = 0.0
    mult1 = 1.0
    mult2 = 1.0
    use_dtw = False
    use_correlation = False
    use_full_output = False
    use_wd = False
    use_ks = False
    use_adf = False
    use_coint = False
    single_flow = None
    flows = None
    display_columns=False
    resample=False
    baseflow=None
    match=False
    add_grouping=False
    cluster_data=None
    cluster_graph=False
    calibrate=False
    CalibrationSamplePeriodicityInMinutes=5
    CalibrationPeriodicityInHours=1
    MinimumFlowToSample=500
    NumberOfSamplesForCalibration=60
    CalibrationDelayAfterRestartInHours=4
    CalibrationOffsetSamplesFromCurrentTime=24
    synchronize_axis=True
    synchronize_index=True
    uselsq=False
    useopt=False
    config=None
    calibration_output=None
    detached = False

class Ui(QtWidgets.QDialog):
    def __init__(self):
        
        super(Ui, self).__init__()
        self.ui=uic.loadUi('flow_analyzer.ui', self)
        self.args = Args()

        self.selectInputFile.clicked.connect(self.onSelectInputFile)
        self.selectOutputFile.clicked.connect(self.onSelectOutputFile)
        self.selectOutputTable.clicked.connect(self.onSelectOutputTable)
        self.selectConfigFile.clicked.connect(self.onSelectConfigFile)
        self.filterFromCheck.clicked.connect(self.onFilterFrom)
        self.filterToCheck.clicked.connect(self.onFilterTo)
        self.processButton.clicked.connect(self.onProcess)
        self.exitButton.clicked.connect(self.onExit)
        self.calibrationSetup.clicked.connect(self.onCalibrationSetup)
        self.useSingleFlow.clicked.connect(self.onUseSingleFlow)
        self.isBaseFlowNeeded.clicked.connect(self.onBaseFlowNeeded)
        self.checkUseCluster.clicked.connect(self.onUseCluster)

        self.filterFrom.setDate(datetime.datetime.now().date())
        self.filterTo.setDate(datetime.datetime.now().date())

        self.show()

    def FileDialog(self,directory='', forOpen=True, fmt='', isFolder=False):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QDir.Hidden)

        # ARE WE TALKING ABOUT FILES OR FOLDERS
        if isFolder:
            dialog.setFileMode(QFileDialog.DirectoryOnly)
        else:
            dialog.setFileMode(QFileDialog.AnyFile)
        # OPENING OR SAVING
        dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

        # SET FORMAT, IF SPECIFIED
        if fmt != '' and isFolder is False:
            dialog.setDefaultSuffix(fmt)
            dialog.setNameFilters([f'{fmt} (*.{fmt})'])

        # SET THE STARTING DIRECTORY
        if directory != '':
            dialog.setDirectory(str(directory))


        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0]  # returns a list
            return path
        else:
            return ''

    def onCalibrationSetup(self):
        calibrate=calibrategui.Calibrate(self.args)
        calibrate.show()

    def saveFileNameDialog(self,ext="Text Files (*.txt);;All Files (*)"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save file","",ext, options=options)
        return fileName

    def openFileNameDialog(self,ext="All Files (*)"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileNames, _ = QFileDialog.getOpenFileNames(self,"Open files", "",ext, options=options)
        return fileNames
    
    def openFolderNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = QFileDialog.getExistingDirectory(self,"Open directory")
        return fileName
    
    def onSelectOutputFile(self,ext="All Files (*)"):
        file=self.saveFileNameDialog("TXT Files (*.txt);;All Files (*)")
        if not file is None and file!="":
            self.outputFile.setText(file)
    
    def onSelectConfigFile(self,ext="JSON files (*.json);;All Files (*)"):
        files=self.openFileNameDialog("JSON Files (*.json);;All Files (*)")
        if files:
            self.configFile.setText(files[0])
 

    def onSelectOutputTable(self,ext="All Files (*)"):
        file=self.saveFileNameDialog("CSV Files (*.csv);;All Files (*)")
        if not file is None and file!="":
            self.outputTable.setText(file)
    
   
    def onSelectInputFile(self,ext="All Files (*)"):
        files=self.openFileNameDialog("CSV Files (*.csv);;All Files (*)")
        if files:
            self.inputFile.setText(files[0])
            if not self.load_flows(files[0]):
                self.inputFile.setText("")
    
    def load_flows(self,fname):
        try:
            df=pd.read_csv(fname,nrows=1)
            df["TIME"]=pd.to_datetime(df["TIME"],format="%Y/%m/%d %H:%M:%S.%f")
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No data")
            msg.setInformativeText('CSV file is invalid or no header found')
            msg.setWindowTitle("Error")
            msg.exec_()

        cols={}
        for c in df.columns:
            i=c.find(':VAL')
            if i!=-1:
                cols[c]=c[:i]
            else:
                cols[c]=c
            cols[c]=cols[c].strip()
        df.rename(columns=cols,inplace=True)
        vars=list(df.columns[1:])
        self.varList.clear()
        self.baseFlow.clear()
        if len(vars)>0:
            self.varList.addItems(vars)
            self.baseFlow.addItems(vars)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("No data")
            msg.setInformativeText('CSV file is invalid or no header found')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        return True


    def onProcess(self):
        if self.inputFile.text()=="" or len(self.varList.selectedIndexes())==0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Input file not given or flow variables are the same")
            msg.setInformativeText('Incorrect or missing information')
            msg.setWindowTitle("Error")
            msg.exec_() 
            return 

        
        self.args.input=self.inputFile.text()
        self.args.output=None if self.outputFile.text()=="" else self.outputFile.text()
        self.args.table=None if self.outputTable.text()=="" else self.outputTable.text()
        self.args.frm=None
        if self.filterFromCheck.isChecked():
            self.args.frm=self.filterFrom.text()
        self.args.to=None
        if self.filterToCheck.isChecked():
            self.args.to=self.filterTo.text()
        self.args.display_charts_1=self.graph1Check.isChecked()
        self.args.display_charts_2=self.graph2Check.isChecked()
        self.args.display_charts_3=self.graph3.currentIndex()
        self.args.display_charts_4=self.graph4Check.isChecked()
        self.args.display_charts_5=self.graph5Check.isChecked()
        self.args.display_grid=self.displayGridCheck.isChecked()
        self.args.abs_value=self.absDiffCheck.isChecked()
        self.args.bias1=self.bias1Value.value()
        self.args.bias2=self.bias2Value.value()
        self.args.mult1=self.mult1Value.value()
        self.args.mult2=self.mult2Value.value()
        self.args.use_dtw=self.dtwCheck.isChecked()
        self.args.use_wd=self.wdCheck.isChecked()
        self.args.use_ks=self.ksCheck.isChecked()
        self.args.use_adf=self.adfCheck.isChecked()
        self.args.detached=self.checkDetached.isChecked()
        self.args.use_full_output=self.fullOutputCheck.isChecked()
        self.args.use_coint=self.cointCheck.isChecked()
        self.args.use_correlation=self.corrCheck.isChecked()
        self.args.add_grouping=self.checkAddGrouping.isChecked()
        if self.checkUseCluster.isChecked():
            self.args.cluster_data=self.clusterValue.value()
            self.args.cluster_graph=self.checkClusterGraph.isChecked()
        else:
            self.args.cluster_data=None
            self.args.cluster_graph=False
        
        if not self.useSingleFlow.isChecked():
            self.args.single_flow=None
            self.args.flows=[x.data() for x in self.varList.selectedIndexes()]
            if self.isBaseFlowNeeded.isChecked():
                self.args.baseflow=[self.baseFlow.currentText()]
        else:
            self.args.flows=None
            self.args.baseflow=None
            self.args.single_flow=[[x.data() for x in self.varList.selectedIndexes()][0]]
        self.args.config=None if self.configFile.text()=="" else self.configFile.text()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fr.main(self.args)
        except Exception as ex:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(str(ex))
            msg.setWindowTitle("Error")
            msg.exec()
        QApplication.restoreOverrideCursor()
        #QApplication.quit()

    def onExit(self):
        QApplication.quit()

    def onFilterFrom(self):
        self.filterFrom.setEnabled(self.filterFromCheck.isChecked())

    def onUseCluster(self):
        self.clusterValue.setEnabled(self.checkUseCluster.isChecked())
        self.checkClusterGraph.setEnabled(self.checkUseCluster.isChecked())

    def onUseSingleFlow(self):
        selmode=QAbstractItemView.MultiSelection
        if self.useSingleFlow.isChecked():
            self.isBaseFlowNeeded.setChecked(False)
            self.baseFlow.setEnabled(False)
            self.isBaseFlowNeeded.setEnabled(False)
            self.bias2Value.setEnabled(False)
            self.mult2Value.setEnabled(False)
            selmode=QAbstractItemView.SingleSelection
        else:
            self.baseFlow.setEnabled(False)
            self.isBaseFlowNeeded.setChecked(False)
            self.isBaseFlowNeeded.setEnabled(True)
            self.mult2Value.setEnabled(True)
            
        self.varList.setSelectionMode(selmode)
        self.varList.clearSelection()

    def onBaseFlowNeeded(self):
        self.baseFlow.setEnabled(self.isBaseFlowNeeded.isChecked() and self.baseFlow.count()>0)

    def onFilterTo(self):
        self.filterTo.setEnabled(self.filterToCheck.isChecked())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())

