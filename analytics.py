import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from music21 import *
from time import sleep
from midi2audio import FluidSynth
import os
import mir_eval
import collections
import math

#install librosa
#install music21
#install midi2audio (fluidsynth)

def analyze_audio(original, recording, tempo):
    o_y, o_sr = librosa.load(original);
    r_y, r_sr = librosa.load(recording);

    # calculate tempos and tempo evaluation (USE GIVEN TEMPO HERE? ALSO REPEAT IS DUMB, probably just change this to my own thing)
    o_tempo, o_beat_frames = librosa.beat.beat_track(y=o_y,sr=o_sr,start_bpm=tempo)
    r_tempo, r_beat_frames = librosa.beat.beat_track(y=r_y,sr=r_sr,start_bpm=tempo)
    o_beats, r_beats = librosa.frames_to_time(o_beat_frames,sr=o_sr), librosa.frames_to_time(r_beat_frames,sr=r_sr)
    ref_weight = 0.5
    # mir_eval.tempo.validate(np.repeat(o_tempo,2),ref_weight,np.repeat(r_tempo,2))
    tempo_score = mir_eval.tempo.evaluate(np.repeat(o_tempo,2), ref_weight, np.repeat(r_tempo,2))['P-score']

    # beat calculations using DP
    o_beats, r_beats= mir_eval.beat.trim_beats(o_beats), mir_eval.beat.trim_beats(r_beats)
    beat_metrics = mir_eval.beat.evaluate(o_beats, r_beats)
    beat_p, beat_kl = beat_metrics['P-score'], beat_metrics['Information gain']

    # onset calculation (between [0,1])
    o_onsets = librosa.onset.onset_detect(y=o_y,sr=o_sr,units='time')
    r_onsets = librosa.onset.onset_detect(y=r_y,sr=r_sr,units='time')
    onset_precision = mir_eval.onset.evaluate(o_onsets,r_onsets)['Precision']

    # cosine similarity between spectral centroids
    o_centroids = librosa.feature.spectral_centroid(y=o_y,sr=o_sr)
    r_centroids = librosa.feature.spectral_centroid(y=r_y,sr=r_sr)
    o_len, r_len = o_centroids.shape[1], r_centroids.shape[1]
    if o_len < r_len:
        r_centroids = r_centroids[:,(r_len-o_len):]
    else:
        o_centroids = o_centroids[:,(o_len-r_len):]
    centroid_sim = np.sum(o_centroids * r_centroids) / (np.linalg.norm(o_centroids) * np.linalg.norm(r_centroids))

    # chroma freq (12 pitch classes per frame, compute freq for max per frame using Short Time FT)
    o_chroma, r_chroma = librosa.feature.chroma_stft(y=o_y,sr=o_sr), librosa.feature.chroma_stft(y=r_y,sr=r_sr)
    o_mchroma, r_mchroma = np.argmax(o_chroma,axis=0), np.argmax(r_chroma,axis=0)
    o_counts, r_counts = collections.Counter(o_mchroma), collections.Counter(r_mchroma)
    oc_len, rc_len = len(o_mchroma), len(r_mchroma)
    if oc_len < rc_len:
        r_mchroma = r_mchroma[(rc_len-oc_len):]
    elif oc_len > rc_len:
        o_mchroma = o_mchroma[(oc_len-rc_len):]
    nmse = 0
    for i in range(len(o_mchroma)):
        if abs(o_mchroma[i] - r_mchroma[i]) > 2:
            nmse += np.sign(o_mchroma[i] - r_mchroma[i])
    nmse /= oc_len
    nmse = 1 - abs(nmse)

    # probabilistic YIN (HMM model on pitch classes for computing fundamental frequencies)
    """
    o_f0, ovf, ovp = librosa.pyin(o_y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    r_f0, rvf, rvp = librosa.pyin(r_y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    o_f0 = np.array([el for el in o_f0 if not math.isnan(el)])
    r_f0 = np.array([el for el in r_f0 if not math.isnan(el)])
    of_len, rf_len = len(o_f0), len(r_f0)
    if of_len < rf_len:
        r_f0 = r_f0[(rf_len-of_len):]
    else:
        o_f0 = o_f0[(of_len-rf_len):]
    f0_sim = np.sum(o_f0 * r_f0) / (np.linalg.norm(o_f0) * np.linalg.norm(r_f0))
    """
    f0_sim = centroid_sim

    # MFCC (mel freq cepstral coefficients for ML)
    return (tempo_score, beat_p, beat_kl, onset_precision, nmse, centroid_sim, f0_sim)

def m21_info(mxml, part, measure_start, measure_end, true_tempo):
    b = converter.parse(mxml)
    trunc_b = b.measures(measure_start,measure_end)
    iso = (trunc_b.parts.stream())[part].flat
    iso.insert(0,tempo.MetronomeMark(number=true_tempo))

    # USE TEMPORARY DIRECTORIES FOR FRAG AND PERFECT.wav

    # mxml -> midi
    frag = iso.write(fmt="midi",fp="frag.midi")

    # midi -> wav
    fs = FluidSynth('def.sf2')
    fs.midi_to_audio("frag.midi",f"{os.path.splitext(mxml)[0]}.wav")

    #time signature, key
    key, ts = iso.analyze('key'), iso.getTimeSignatures()[0].ratioString

    # pitchAnalysis
    # pcCount = analysis.pitchAnalysis.pitchAttributeCount(iso, 'pitchClass')
    return key, ts


def analyze(mxml, part, measure_start, measure_end, tempo, recording):
    # translate mxml -> midi -> audio
    key, ts = m21_info(mxml, part, measure_start, measure_end, tempo)
    perfect = f"{os.path.splitext(mxml)[0]}.wav"
    tempo_score, beat_p, beat_kl, onset_precision, chroma_info, centroid_sim, f0_sim = analyze_audio(perfect,recording,tempo)

    tb_score, freq_score, pitch_score = tempo_score / 18.0 + (beat_p + beat_kl) / 9.0 + onset_precision / 6.0, (centroid_sim + f0_sim) / 6.0, 2.0 * chroma_info / 9.0
    similarity_score = tb_score + freq_score + pitch_score

    print(f"SIM: {similarity_score}, TEMPO + BEAT: {9.0 * tb_score / 4.0}, FREQ: {freq_score * 3.0}, PITCH: {chroma_info}")
    return similarity_score, 9 * tb_score / 4.0 , 3.0 * freq_score, chroma_info

if __name__ == "__main__":
    analyze('tlby.musicxml','Alto', 1, 30, 78, "alto_church.wav")
    analyze('tlby.musicxml','Tenor', 1, 30, 78, "tenor_church.wav")
    analyze('guitar.musicxml','Guitar', 1, 30, 84, "tenor_church.wav")
    analyze('guitar.musicxml','Guitar', 1, 30, 84, "guitar.wav")
