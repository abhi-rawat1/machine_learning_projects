import pysrt
import os
from os import path
import soundfile as sf
import numpy as np
import pandas as pd
import subprocess

def timeToSec(timeVal):
    timeToSec = timeVal.hour*3600 + timeVal.minute*60 + timeVal.second
    return timeToSec

def getAudioDuration(audioFilePath):
    f = sf.SoundFile(audioFilePath)
    noOfSamples = len(f)
    sampleRate = f.samplerate
    noOfSeconds = int(noOfSamples/sampleRate)
    return noOfSeconds

basepath = path.dirname(__file__)
videoDirPath = os.path.join(basepath,'videos1')

for filename in os.listdir(videoDirPath):
    if filename.endswith(".srt"):
        print('processing ' + filename)
        input_srt_file = os.path.join(videoDirPath, filename)
        fileNameWithOutExt = os.path.splitext(filename)[0]
        audioFilePath = os.path.join(videoDirPath, fileNameWithOutExt+'.flac')
        totalDuration = getAudioDuration(audioFilePath)
        subTitleIndexList = np.arange(0,totalDuration)
        subTitleLabelList = [0] * totalDuration
        subTitleDataFrame = pd.DataFrame(list(zip(subTitleIndexList, subTitleLabelList)), columns=['fileIndex', 'label'])
        subTitleDataFrame = subTitleDataFrame.astype(int)
        subTitleDataFrame['fileIndex'] = fileNameWithOutExt + '_' + subTitleDataFrame['fileIndex'].astype(str)
        subTitlePositiveLabelSet = set()
        sub = pysrt.open(input_srt_file)
        # Start and End time
        for sub_record in sub:
            start = sub_record.start.to_time()
            startSec = timeToSec(start) - 1 # -1 offset, as melspectrogram images start with 0 index
            end = sub_record.end.to_time()
            endSec = timeToSec(end) - 1 # -1 offset, as melspectrogram images start with 0 index
            subRecordSecRange = np.arange(startSec, endSec+1)
            subTitlePositiveLabelSet.update(subRecordSecRange)
        subTitleDataFrame.loc[subTitlePositiveLabelSet, 'label'] = 1
        subTitleDataFrame.to_csv('features1.csv', header=False, mode='a')
print('done')