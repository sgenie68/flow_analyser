from datetime import datetime, timedelta
import numpy as np
import argparse


def DoubleSVDLeastSq(A,b):
    # Perform Singular Value Decomposition (SVD)
    print(A.shape)
    print(b.shape)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(U.shape)
    print(S)

    # Compute pseudo-inverse of S
    S_pseudo_inv = np.diag(1 / S)  # Pseudo-inverse of S (diagonal matrix)

    # Solve least squares using pseudo-inverse
    x = Vt.T @ S_pseudo_inv @ U.T @ b
    print(x)
    return x

class CalibrationSystem:
    def __init__(self,arg):
        self.MinimumFlowToSample = arg.MinimumFlowToSample
        self.CalibrationDelayAfterRestart = timedelta(hours=arg.CalibrationDelayAfterRestartInHours)  # Example value
        self.CalibrationSamplePeriodicity = timedelta(minutes=arg.CalibrationSamplePeriodicityInMinutes)  # Example value
        self.CalibrationPeriodicity = timedelta(hours=arg.CalibrationPeriodicityInHours)  # Example value
        self.NumberOfSamplesForCalibration = arg.NumberOfSamplesForCalibration  # Example value
        self.CalibrationOffsetFromCurrentSample = arg.CalibrationOffsetSamplesFromCurrentTime  # Example value

        self._isShutdown = True
        self._isWaitingForRestartDelayToExpire = False
        self._timeOfLastRestart = None
        self._lastSampleTime = None
        self._lastCalibrationSampleTime = None
        self._lastCalibrationTime = None
        self._packAtLastCalibrationSample = 0
        self._inletFlowAccumulationSinceLastCalibrationSample = 0
        self._outletFlowAccumulationSinceLastCalibrationSample = 0
        self._lastInletFlow = 0
        self._lastOutletFlow = 0

        self.InletFlowBufferedList = []
        self.OutletFlowBufferedList = []
        self.PackRateBufferedList = []

        self._A = None
        self._b = None
        self.InletFlowMultiplier = 1
        self.firstUpdate=False

    def add_sample_and_calibrate(self, current_time, inlet_flow, outlet_flow, pack):
        is_currently_shutdown = (inlet_flow < self.MinimumFlowToSample or 
                                 outlet_flow < self.MinimumFlowToSample)

        if not is_currently_shutdown and self._isShutdown:
            self._timeOfLastRestart = current_time
            self._isWaitingForRestartDelayToExpire = True
            self._isShutdown = False
            self._lastSampleTime = current_time

        if self._lastSampleTime is None:
            self._lastSampleTime = current_time
            self._lastCalibrationSampleTime = current_time
            self._packAtLastCalibrationSample = pack
            self._isWaitingForRestartDelayToExpire = True

        if is_currently_shutdown:
            self._inletFlowAccumulationSinceLastCalibrationSample = 0
            self._outletFlowAccumulationSinceLastCalibrationSample = 0
            self._packAtLastCalibrationSample = pack
            self._lastSampleTime = current_time
            self._isShutdown = True
            return False

        if self._isWaitingForRestartDelayToExpire:
            if current_time - self._timeOfLastRestart < self.CalibrationDelayAfterRestart:
                return False

            self._lastSampleTime = current_time
            self._lastCalibrationSampleTime = current_time
            self._inletFlowAccumulationSinceLastCalibrationSample = 0
            self._outletFlowAccumulationSinceLastCalibrationSample = 0
            self._packAtLastCalibrationSample = pack
            self._lastInletFlow = inlet_flow
            self._lastOutletFlow = outlet_flow
            self._isWaitingForRestartDelayToExpire = False
            return False

        hours_since_last_sample = (current_time - self._lastSampleTime).total_seconds() / 3600
        self._inletFlowAccumulationSinceLastCalibrationSample += self._lastInletFlow * hours_since_last_sample
        self._outletFlowAccumulationSinceLastCalibrationSample += self._lastOutletFlow * hours_since_last_sample

        self._lastInletFlow = inlet_flow
        self._lastOutletFlow = outlet_flow
        self._lastSampleTime = current_time

        time_span_since_last_calibration_sample = current_time - self._lastCalibrationSampleTime
        if time_span_since_last_calibration_sample >= self.CalibrationSamplePeriodicity:
            hours_since_calibration_sample = time_span_since_last_calibration_sample.total_seconds() / 3600
            self.InletFlowBufferedList.append(self._inletFlowAccumulationSinceLastCalibrationSample / hours_since_calibration_sample)
            self.OutletFlowBufferedList.append(self._outletFlowAccumulationSinceLastCalibrationSample / hours_since_calibration_sample)
            self.PackRateBufferedList.append((pack - self._packAtLastCalibrationSample) / hours_since_calibration_sample)

            self._packAtLastCalibrationSample = pack
            self._inletFlowAccumulationSinceLastCalibrationSample = 0
            self._outletFlowAccumulationSinceLastCalibrationSample = 0
            self._lastCalibrationSampleTime = current_time

            if len(self.InletFlowBufferedList) > self.NumberOfSamplesForCalibration + self.CalibrationOffsetFromCurrentSample:
                self.InletFlowBufferedList.pop(0)
                self.OutletFlowBufferedList.pop(0)
                self.PackRateBufferedList.pop(0)
        else:
            return False

        if (len(self.InletFlowBufferedList) < self.NumberOfSamplesForCalibration + self.CalibrationOffsetFromCurrentSample or 
                (current_time - self._lastCalibrationTime if self._lastCalibrationTime else timedelta.max) < self.CalibrationPeriodicity):
            return False

        if self._A is None:
            self._A = np.zeros((self.NumberOfSamplesForCalibration,2))
            self._b = np.zeros(self.NumberOfSamplesForCalibration)

        flow_non_zero = False
        for i in range(len(self.InletFlowBufferedList) - self.CalibrationOffsetFromCurrentSample):
            if abs(self.InletFlowBufferedList[i]) > 1e-6:  # Replace 1e-6 with appropriate tolerance
                flow_non_zero = True

            self._A[i, 0] = self.InletFlowBufferedList[i]
            self._b[i] = (self.PackRateBufferedList[i] + self.OutletFlowBufferedList[i])

        if not flow_non_zero:
            return False

        try:
            x, _, _, _ = np.linalg.lstsq(self._A, self._b, rcond=None)
            self.InletFlowMultiplier = x[0]
            self.firstUpdate=True
        except Exception as e:
            raise RuntimeError(f"CalibrateInletFlowAgainstOutletFlow threw exception {str(e)}")

        self._lastCalibrationTime = current_time
        return True




