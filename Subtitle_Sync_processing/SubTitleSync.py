'''
This python file is trying to sync given subtitle (SRT) file with given video file based on determining non-mute sections in video file.

Steps to sync SRT file are:
1. Split original video file into chunk of 10 min video [using ffmpeg]
2. Extract Audio signal from each chunk video file [using ffmpeg]
3. Extract Vocal signals from each audio file [using spleeter]
4. Get not-mute sections from each voca signal using Pydub library
5. Overlap not-mute sections based on given threshold value 
6. Read SRT file and trasnform it into JSON structure
7. Sync subtitles based on non-mute sections
8. Generate new synced subtitle file 

'''

#!/usr/bin/python3
import argparse
import numpy as np
import os
from os import path
import subprocess
import shutil
from pydub import AudioSegment
from pydub.silence import  detect_nonsilent
import datetime
import copy
import pysrt
import time
import pandas as pd

def split_video(input_video_file, outputVideoSplitFileNames):
    print('in the split audio: ', input_video_file)
    extract_sound_cmd = [
         'ffmpeg',
         '-y',  # overwrite if exists
         '-i', input_video_file,
         '-c', 'copy',
         '-map', '0',
         '-segment_time', '00:10:00',
         '-f', 'segment',
         '-reset_timestamps', '1',
         '-loglevel', 'error',
         outputVideoSplitFileNames
     ]
    output = subprocess.call(extract_sound_cmd)
    print('Split video porcess completed with code: ' ,output)
    

def extractAudio(videoDirPath, outputAudeoDirPath):
    for filename in os.listdir(videoDirPath):
        if filename.endswith(".mp4"):
            input_video_file = os.path.join(videoDirPath, filename)
            fileNameWithOutExt = os.path.splitext(filename)[0]
            output_sound_file = os.path.join(outputAudeoDirPath, fileNameWithOutExt+'.flac')
            print(os.path.join(videoDirPath, filename))
            extract_sound_cmd = [
                 'ffmpeg',
                 '-y',  # overwrite if exists
                 '-loglevel', 'error',
                 '-i', input_video_file,  # input
                 '-ac', '1',  # convert to mono
                 output_sound_file
             ]
            subprocess.call(extract_sound_cmd)
    
    print('Extracting Audios from Videos has been completed')


def extractVocalSignals(outputAudeoDirPath):
    for filename in os.listdir(outputAudeoDirPath):
        input_audio_file = os.path.join(outputAudeoDirPath, filename)
        extract_sound_cmd = [
             'spleeter',
             'separate',  # overwrite if exists
             '-i', input_audio_file,  # input
             '-p', 'spleeter:2stems',  # convert to mono
            '-d', '600',
            '-o', outputAudeoDirPath
         ]
        output = subprocess.call(extract_sound_cmd)
        print('output: ', output)
    

def getNonMuteSections(outputAudeoDirPath):
    nonMuteSections = []
    for filename in os.listdir(outputAudeoDirPath):
        if ".flac" not in filename:
            vocalDir = os.path.join(outputAudeoDirPath, filename)
            for vocalFileName in os.listdir(vocalDir):
                if vocalFileName == 'vocals.wav':
                    vocalFile = os.path.join(vocalDir, vocalFileName)
                    audio_signal = AudioSegment.from_file(vocalFile, "wav")
                    print(vocalDir, '  Duration of Audio Signal: ', len(audio_signal) / 1000)
                    nonsilent_audio_range = detect_nonsilent(audio_signal,min_silence_len=1000,silence_thresh=-33)
                    nonMuteSections.append(nonsilent_audio_range)
    
    return nonMuteSections[0]
    
def overlapMuteSections(nonMuteSections):
    previous_end_ts = 0
    modified_audio_sections = []
    index = -1
    for section in nonMuteSections:
        if index == -1:
            modified_audio_sections.append([section[0], section[1]])
            index = index+1
        else:
            if (section[0] - modified_audio_sections[index][1]) <= 1000: 
                modified_audio_sections[index][1] = section[1]
            else:
                modified_audio_sections.append([section[0], section[1]])
                index = index+1
    
    return modified_audio_sections

def extractSubTitleDetails(subs):
    transofrmed_subs = []
    index = 1
    for sub_section in subs:

        startTimeStamp = ((int(sub_section.start.hours)*60 + int(sub_section.start.minutes))*60 + int(sub_section.start.seconds))*1000
        endTimeStamp = ((int(sub_section.end.hours)*60 + int(sub_section.end.minutes))*60 + int(sub_section.end.seconds))*1000
        text = sub_section.text
        transofrmed_subs.append([index, startTimeStamp, endTimeStamp, text])
        index = index+1
    
    start_timestamp_srt = transofrmed_subs[0][1]
    end_timestamp_srt = transofrmed_subs[len(transofrmed_subs) -1 ][1]
    return transofrmed_subs, start_timestamp_srt, end_timestamp_srt
    
def subTitleSync(transofrmed_subs, nonsilent_audio_range):
    non_mute_index = 0
    sub_section_index = -1
    for sub_section in transofrmed_subs:
        sub_section_index = sub_section_index + 1
        start_ts_srt = sub_section[1]
        end_ts_srt = sub_section[2]
        
        st = 0
        et = 0
        st_received = False
        et_received = False
        for i in range(non_mute_index, len(nonsilent_audio_range)):
            non_mute_index = non_mute_index + 1
            start_ts_audio = nonsilent_audio_range[i][0]
            end_ts_audio = nonsilent_audio_range[i][1]
            
            if not st_received and ((end_ts_srt > start_ts_audio+1000 >= start_ts_srt) or (start_ts_srt < start_ts_audio < end_ts_srt)):
                st_received = True
                st = start_ts_audio
                
            if (start_ts_srt < end_ts_audio <= end_ts_srt):
                et = end_ts_audio
            elif (end_ts_audio-1000 <= end_ts_srt):
                et = end_ts_audio
                et_received = True
            
            if end_ts_audio-1000 > end_ts_srt:
                et_received = True
                non_mute_index = non_mute_index - 1
                
            if et_received and st_received:
                
                if st != 0 and et!= 0 and st > transofrmed_subs[sub_section_index-1][2]:
                    transofrmed_subs[sub_section_index][1] = st
                    transofrmed_subs[sub_section_index][2] = et
                
                #print(sub_section_index, ' : ', transofrmed_subs[sub_section_index][1], ' : ', transofrmed_subs[sub_section_index][2])
                break;
    
    return transofrmed_subs


def generateSyncSubTitle(subs_copy, transofrmed_subs, modified_srt_file):
    index = 0
    for sub_section in subs_copy:
        start_ts = (pd.to_datetime(transofrmed_subs[index][1], unit='ms').strftime('%H:%M:%S:%f')).split(':')
        end_ts = (pd.to_datetime(transofrmed_subs[index][2], unit='ms').strftime('%H:%M:%S:%f')).split(':')
        
        sub_section.start.milliseconds = int(start_ts[3][0:3])
        sub_section.start.seconds = int(start_ts[2])
        sub_section.start.minutes = int(start_ts[1])
        sub_section.start.hours = int(start_ts[0])
        
        sub_section.end.milliseconds = int(end_ts[3][0:3])
        sub_section.end.seconds = int(end_ts[2])
        sub_section.end.minutes = int(end_ts[1])
        sub_section.end.hours = int(end_ts[0])
        
        index = index +1
    
    subs_copy.save(modified_srt_file, encoding='utf-8')


basepath = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(basepath,'data')
input_video_fie_name = 'GardenofEvil.mp4'
input_srt_file_name = 'GardenofEvil.srt'
modified_srt_file_name = 'GardenofEvil_Sync.srt'

input_video_file = os.path.join(data_path, input_video_fie_name)
input_srt_file = os.path.join(data_path, input_srt_file_name)
modified_srt_file = os.path.join(data_path, modified_srt_file_name)

outputDirPath = os.path.join(data_path, input_video_fie_name.split('.')[0]) 

outputVideoDirPath = os.path.join(outputDirPath, 'video_split')
outputVideoSplitFileNames = os.path.join(outputVideoDirPath, 'video%03d.mp4')

outputAudeoDirPath = os.path.join(outputDirPath, 'audio')


shutil.rmtree(outputDirPath, ignore_errors=True)
os.makedirs(outputDirPath) 
os.makedirs(outputVideoDirPath)
os.makedirs(outputAudeoDirPath)

print('Step-1 of 8: Splitting Videos in chunk of 10 mins.')
split_video(input_video_file,outputVideoSplitFileNames)

print('Step-2 of 8: Extracting Audio from each splitted Video.')
extractAudio(outputVideoDirPath, outputAudeoDirPath)

print('Step-3 of 8: Extracting Vocal Signals from each Audio.')
extractVocalSignals(outputAudeoDirPath)

print('Step-4 of 8: Extracting Non mute sections from Vocal Signals.')
nonMuteSections = getNonMuteSections(outputAudeoDirPath)

print('Step-5 of 8: Otimizing Non mute setions.')
nonMuteSections = overlapMuteSections(nonMuteSections)
start_timestamp_audio = nonMuteSections[0][0]
end_timestamp_audio = nonMuteSections[len(nonMuteSections)-1][0]

print('Lenght of NonMute Sections: ', len(nonMuteSections))

subs = pysrt.open(input_srt_file)
subs_copy=  copy.deepcopy(subs)

print('Step-6 of 8: Extracting Subtitle timestamp details from given subtitle file.')
transofrmed_subs, start_timestamp_srt, end_timestamp_srt = extractSubTitleDetails(subs)

print('Step-7 of 8: Syncing Subtitle timestamp details based on non-mute sections.')
sync_subs = subTitleSync(transofrmed_subs, nonMuteSections)

print('Step-8 of 8: Generating new sync Subtitle file.')
generateSyncSubTitle(subs_copy, sync_subs, modified_srt_file)

print('Resulted Sync file is available at: ', modified_srt_file)