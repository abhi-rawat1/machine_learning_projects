Basic wave/sound wave concpet:  https://medium.com/game-of-bits/audio-analysis-part-1-sound-waves-things-you-must-know-1e10851cc109

Audio processing in python: https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

What are the audio features such as MFCC, GFCC and how to use them for Audio Classification:
	https://opensource.com/article/19/9/audio-processing-machine-learning-python
	
Various ways to perform audio classification:
a. Use Audio Numeric Features like mfcc, zcr etc and train model using ML algorithm like SVM, LR etc. 
b. Use Audio Numeric Features like mfcc, zcr etc and train model using Neural Network (Keras Sequnetial Model)
https://medium.com/@sdoshi579/classification-of-music-into-different-genres-using-keras-82ab5339efe0	
c. Use MelSpectogram or Spectogram images and train model using deep learning -
	https://towardsdatascience.com/sound-classification-using-images-68d4770df426
d. To perform deep learning based model, there are different ways to train model:
	1. Full training where built all layers and train model end to end
	2. Transfer Learning (can use different model such as resnet, mobilenet, VGG16 etc) - https://www.pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/
	3. Transfer Learning with fine tuning - https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/



For CNN Model callback - earlyStopping and check point
	https://github.com/tangming2008/Projects_Metis/blob/master/5_Apply%20Deep%20Learning%20to%20Detect%20Blurry%20Images/code/3_model_CNN.ipynb


Image Quality Brisque Score: https://www.learnopencv.com/image-quality-assessment-brisque/

video Quality Netflix VMAF Score : 
	https://netflixtechblog.com/vmaf-the-journey-continues-44b51ee9ed12
	https://github.com/Netflix/vmaf

Video Quality Score such as NIQE, PSNR, RECO, SSIm 
	https://github.com/aizvorski/video-quality

Audio Classification using MelSpectogram images:  https://towardsdatascience.com/sound-classification-using-images-68d4770df426

10 Audio Processing use cases with example: https://www.analyticsvidhya.com/blog/2018/01/10-audio-processing-projects-applications/
	Music Tagging, Music Separation, Music Recommendation 
	
Time - time of an audio 
Amplitude - Amplitude represents the magnitude of the wave signal and it is usually measured in decibels (dB).
Frequency - represents how many complete cycle the wave takes in one second and it is measured in Hz
Sampling Rate - number of samples (amplitudes) per soconds in audio file. 
				if sampling rate is 16K that means while recording this file we were capturing 16000 amplitudes every second. 
				idelly it is kept 2 times of frequencies 
				As human hearing range is around 20K Hz, sampling rate of audio files in many libraries are by default set at 40K per sec. 
Frame - a window of 20-3o ms in a audio signal 
Duration - Audio total duration. To calcualte audio duration, we can simply divide the total number of samples (amplitudes) present in audio by the sampling-rate

Sound wave: It will have all above audio features which can be extracted. 
Audio signal: Audio signal can consitute one or more than one sound wave. Combination of all sound wave rsult in final audio signal.

waveplot -> graph between time and amplitude. Tis is known as time domain representation
Fast Fourier Transformation - known as frequency domain. Plot between amplitude and frequency
Spectogram - plot between time, frequency and amplitude
MFCC — Mel-Frequency Cepstral Coefficients


To add noise in Audio signal:  https://github.com/sleekEagle/audio_processing

Urban Sound Data Challange Notebook - 
	a. https://github.com/mikesmales/Udacity-ML-Capstone
	b. https://github.com/AmritK10/Urban-Sound-Classification
	c. https://github.com/aqibsaeed/Urban-Sound-Classification  (most popular)
	
Stream Sound Classification= https://github.com/chathuranga95/SoundEventClassification

Download Free Noise Sound:https://www.partnersinrhyme.com/soundfx/electricsoundfx.shtml  && https://noises.online/
Possible type of Noise in audio: https://blog.accusonus.com/fix-broken-audio/types-of-noise

Download sound sample - https://philharmonia.co.uk/resources/sound-samples/