# --------------------------------------------------------
# transform audio to text
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------
from moviepy.editor import *
import subprocess
import os
import wave
import numpy as np
import pylab as plt
from model.models import speech_model
import torchaudio as ta
import torch
import time
import shutil

CutTimeDef=15

'''
take the audio file path, and cut it into pieces of the length CutTimeDef, 
then save cutted audios in resdir
'''
def CutFile(files,CutTimeDef,resdir):
    FileName = files
    resdir+=FileName.rstrip('.wav')
    print("CutFile File Name is ",FileName)
    f = wave.open(r"" + FileName, "rb")
    params = f.getparams()
    print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    CutFrameNum = framerate * CutTimeDef
     # read in format info
    print("CutFrameNum=%d" % (CutFrameNum))
    print("nchannels=%d" % (nchannels))
    print("sampwidth=%d" % (sampwidth))
    print("framerate=%d" % (framerate))
    print("nframes=%d" % (nframes))
    str_data = f.readframes(nframes)
    f.close()# wave data to tuple
    # Cutnum =nframes/framerate/CutTimeDef
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    temp_data = wave_data.T
    # StepNum = int(nframes/200)
    StepNum = CutFrameNum
    StepTotalNum = 0
    haha = 0
    while StepTotalNum < nframes:
        print("Stemp=%d" % (haha))
        FileName = os.path.join(files.split('.')[0] +"_"+ str(haha+1) + ".wav")
        print(FileName)
        temp_dataTemp = temp_data[StepNum * (haha):StepNum * (haha + 1)]
        haha = haha + 1
        StepTotalNum = haha * StepNum
        temp_dataTemp.shape = 1, -1
        temp_dataTemp = temp_dataTemp.astype(np.short)# open .wav file
        f = wave.open(FileName, "wb")
        # set attributes of output file
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
         # convert wave data to binary and write into file
        f.writeframes(temp_dataTemp.tostring())
        f.close()

# helper func for converting audio by the model
def get_fu(path_):
    _wavform, _ = ta.load_wav( path_ )
    _feature = ta.compliance.kaldi.fbank(_wavform, num_mel_bins=40) 
    _mean = torch.mean(_feature)
    _std = torch.std(_feature)
    _T_feature =  (_feature - _mean) / _std
    inst_T = _T_feature.unsqueeze(0)
    return inst_T

'''
take the path of audio file, then use loaded model to convert it to text
'''
def get_transp(path_,model_lo,num_wor):
    inst_T = get_fu( path_ )
    log_  = model_lo( inst_T )
    _pre_ = log_.transpose(0,1).detach().numpy()[0]
    liuiu = [dd for dd in _pre_.argmax(-1) if dd != 0]
    str_end = ''.join([ num_wor[dd] for dd in liuiu ])
    return str_end

'''
take path of a video file, split the audio out and cut the audio into pieces,
then convert each cutted piece of audio to text
output whole speech from the video
'''
def audio_to_text(video_path):
    # Split audio out of video, change sample rate, and cut audio for audio2txt model input
    tmpdir='tmp_dir'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    video = VideoFileClip(video_path)
    video_name=video_path.split('/')[-1]
    audio = video.audio
    audio_f=os.path.join(tmpdir,video_name.split('.')[0]+'.wav')
    audio.write_audiofile(audio_f)

    audio_res=os.path.join(tmpdir,video_name.split('.')[0]+'-16k.wav')
    subprocess.call(["sox {} -r 16000 -b 16 -c 1 {}".format(audio_f,audio_res)], shell=True)

    CutFile(audio_res,CutTimeDef,tmpdir)

    # load pre-trained audio2txt model
    model_lo = speech_model()
    device_ = torch.device('cpu')
    model_lo.load_state_dict(torch.load('models/sp_model.pt' , map_location=device_))
    model_lo.eval()

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    num_wor = np.load('models/dic.dic.npy').item()
    # print('model loaded')

    inputf=audio_res.split('.')[0]
    res=''
    filenames = os.listdir(tmpdir)
    cnt=0
    for f in filenames:
        if video_name.split('.')[0] in f and f[0]!='.':
            cnt+=1
    print('cnt',cnt)
    for i in range(cnt):
        try:
            f=inputf+'_'+str(i+1)+'.wav'
            print(f)
            tmpres = get_transp(f,model_lo,num_wor)
            print(tmpres)
            res+=tmpres
        except:
            continue
    shutil.rmtree(tmpdir)
    return res

# if __name__ == "__main__":
#     res=audio_to_text('$DATA/2020-8-14-38.mov')
#     print(res)
