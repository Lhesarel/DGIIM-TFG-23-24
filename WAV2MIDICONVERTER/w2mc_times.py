# WAV TO MIDI CONVERTER (Elapsed time for array of window freqs)
# Javier Granados López
# TFG 2023/2024

# --------------------------Imports--------------------------
from timeit import default_timer as timer
import sys
from scipy.io import wavfile
import os
import numpy as np
from scipy.signal import find_peaks,correlate
import pandas as pd
from sklearn.cluster import KMeans

from music21 import *

import tkinter as tk
from tkinter import messagebox
from PIL import Image,ImageTk

# --------------------------Inputs--------------------------
AUDIO_FILE = 'DoM-piano.wav'                                         # Audio file's name
WINDOW_SIZE_SECS = 0.2                                               # Size of the fft window in seconds
OVERLAPPING_SECS = WINDOW_SIZE_SECS / 2.0                            # Window's overlapping in seconds
SILENCE_THRESHOLD = 0.0001                                           # Intensity threshold for silence in [0,1]
INTENSITY_THRESHOLD = 0.001                                          # Intensity (relevance) threshold for frequencies
MAX_REL_PEAKS = 12                                                   # Maximum number of peaks in the cluster of relevant peaks
MAX_KM_ITERATIONS = 20                                               # Maximum number of K-Means iterations
N_NOTES_CORRECTION_L = 4                                             # Number of notes to the left to consider for correcting a note
N_NOTES_CORRECTION_R = 0                                             # Number of notes to the right to consider for correcting a note
MAX_CORRECTIONS = 1                                                  # Maximum number of corrections

QUARTER_PPM = 62                                                     # Tempo in terms of a quarter
QUANTIZATION_FACTOR = 4                                              # Briefest musical figure to consider

SIGNATURE_NUM = 4                                                    # Numerator of the signature
SIGNATURE_DEN = 4                                                    # Denominator of the signature
TITLE = 'Partitura'                                                  # Title of the piece
COMPOSER = 'WAV2MIDIConverter'                                       # Composer of the piece
FILENAME = 'partitura'                                               # The pdf file's name

# ---------------------Global variables---------------------
WDIMENSIONS = {'little': '750x500','large': '900x750'}
DEFAULTS = {                                                         # Default values 
    "AUDIO_FILE" : AUDIO_FILE,
    "WINDOW_SIZE_SECS": WINDOW_SIZE_SECS,
    "OVERLAPPING_SECS": OVERLAPPING_SECS,
    "SILENCE_THRESHOLD": SILENCE_THRESHOLD,
    "INTENSITY_THRESHOLD": INTENSITY_THRESHOLD,
    "MAX_REL_PEAKS": MAX_REL_PEAKS,
    "MAX_KM_ITERATIONS": MAX_KM_ITERATIONS,
    "N_NOTES_CORRECTION_L": N_NOTES_CORRECTION_L,
    "N_NOTES_CORRECTION_R": N_NOTES_CORRECTION_R,
    "MAX_CORRECTIONS": MAX_CORRECTIONS,
    "QUARTER_PPM": QUARTER_PPM,
    "QUANTIZATION_FACTOR": QUANTIZATION_FACTOR,
    "SIGNATURE_NUM": SIGNATURE_NUM,
    "SIGNATURE_DEN": SIGNATURE_DEN,
    "TITLE": TITLE,
    "COMPOSER": COMPOSER,
    "FILENAME": FILENAME
}
PATH = os.path.dirname(os.path.realpath(sys.argv[0]))                   # Path of project's directory
NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]       # The twelve notes' names

SAMPLE_RATE = None                                                   # Sample rate (samples per second)
SIGNAL = None                                                        # Signal data
WINDOW_SIZE_SAMPLES = None                                           # Size of the fft window in samples
OVERLAPPING_SAMPLES = None                                           # Size of overlapping in samples
AUDIO_SIZE_SECS = None                                               # Size of the audio file in seconds

QUARTER_SECS = None                                                  # Seconds of a quarter

# -------------------------Functions-------------------------
# Notes' detection

def freq_to_number(f):                                                      # Transforms any note's frequency into its midi number 
    return 69 + 12*np.log2(f/440.0)    

def number_to_freq(n):                                                      # Transforms any note's midi number into its frequency
    return 440 * 2.0**((n-69)/12.0)

def note_name(n):                                                           # Gets the note's name given its midi number
    return NOTES[n % 12] + str(int(n/12 - 1))

def extract_window(audio, window_number):                                   # Returns samples of window number <window-number> and true or false whether it's the last window 
    begin = window_number * (WINDOW_SIZE_SAMPLES - OVERLAPPING_SAMPLES)
    end = begin + WINDOW_SIZE_SAMPLES
    
    if end < len(SIGNAL): # Commonly
        return False, audio[begin:end]
    else: # The window surpasses the audio data => Complete last elements of the window with zeros
        return True, np.concatenate([audio[begin:len(SIGNAL)-1],np.zeros(end-len(SIGNAL)+1,dtype=float)])
    
def autocorrelation(window):                                                # Autocorrelation of a given window
    ac = correlate(window,window,mode='full')
    return ac[int(len(ac)/2):]
    
def indexes(freqs,i1,i2,harmonic):                                          # Returns h1 and h2 indexes of the nearest two 
    if i2-i1 == 1:                                                          #     harmonics of window's fund. to harmonic or
        if harmonic == freqs[i1]:                                           #     h1 the index of harmonic in freqs and h2<0
            return i1,-1
        elif harmonic == freqs[i2]:
            return i2,-1
        else:
            return i1,i2
    else:
        isplit = int(i1 + np.ceil((i2-i1)/2.0))
        if harmonic < freqs[isplit]:
            return indexes(freqs,i1,isplit,harmonic)
        elif harmonic > freqs[isplit]:
            return indexes(freqs,isplit,i2,harmonic)
        else:
            return isplit,-1
        
def remove_duplicates(seq): # Remove duplicates preserving order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def detect_peaks(freqs,F): # Returns the array of freqs where fft has relevant peaks
    peaks_before = []
    peaks_after = []
    pindex = find_peaks(F)
    if not len(pindex[0]):
        return []
    num_clusters = 2
    nIterations = 0
    P = pd.DataFrame(data=[F[i] for i in pindex[0]],index=pindex[0],columns=['Intensity'])
    
    kmeans = KMeans(n_clusters=num_clusters,n_init=5,random_state=123456)
    clusters = kmeans.fit_predict(P) # Detect two clusters: peaks and non peaks
    cluster_id = clusters[np.argmin([F[i] for i in pindex[0]])] # Cluster of non relevant peaks id
    rpindex = np.where(clusters != cluster_id)[0] # Indexes of relevant peaks
    peaks_after = [freqs[i] for i in [pindex[0][j] for j in rpindex]]
    num_clusters += 1
    peaks_before = peaks_after.copy()
    while len(peaks_after) <= MAX_REL_PEAKS and nIterations < MAX_KM_ITERATIONS:
        peaks_before = peaks_after.copy()
        nIterations += 1
        kmeans = KMeans(n_clusters=num_clusters,n_init=5,random_state=123456)
        clusters = kmeans.fit_predict(P) # Detect <num_clusters> clusters
        cluster_id = clusters[np.argmin([F[i] for i in pindex[0]])] # Cluster of non relevant peaks id
        rpindex = np.where(clusters != cluster_id)[0] # Indexes of relevant peaks
        peaks_after = [freqs[i] for i in [pindex[0][j] for j in rpindex]]
        num_clusters += 1
        
    return peaks_before

def find_candidates(cset): # Finds candidates for fundamental by substracting elements in cset
    aux_cset = [c for c in cset if c >= 27.5].copy() # Remove too low freqs
    aux_cset.sort() # Order
    candidates = aux_cset.copy()
    for i in range(0,len(aux_cset)-1):
        candidate = number_to_freq(int(round(freq_to_number(aux_cset[i+1] - aux_cset[i])))) # Round to equal temperament
        if candidate not in candidates:
            candidates.append(candidate)
    return [c for c in candidates if c >= 27.5] # Remove too low freqs

def count_harmonics(peaks,candidates,m): # Count the number of harmonics in peaks for each candidate
    nharmonics = np.zeros(len(candidates),dtype=float)
    for i in range(0,len(candidates)):
        for j in range(0,len(peaks)):
            if peaks[j] >= candidates[i]:
                div = np.modf(peaks[j]/candidates[i])[0]
                if np.abs(div-round(div)) < 0.01: # Check if harmonic
                    nharmonics[i] += 1
    for i in range(0,len(candidates)):
        samples = max(1,round(m/candidates[i]))
        nharmonics[i] /= np.power(samples,0.1)
        
    return nharmonics

def count_differences(peaks): # Count the number of harmonics in peaks for each candidate
    aux_peaks = peaks.copy()
    aux_peaks.sort()
    aux_peaks.insert(0,0.0);
    differences = []
    counts = []
    
    for i in range(0,len(aux_peaks)):
        for j in range(i+1,len(aux_peaks)):
            diff = number_to_freq(int(round(freq_to_number(aux_peaks[j] - aux_peaks[i]))))
            if diff not in differences:
                differences.append(diff)
                counts.append(1)
            else:
                counts[differences.index(diff)] += 1
        
    return differences[np.argmax(counts)]

def max_amplitude(fund,freqs,F): # Compute the maximum of amplitudes in fund harmonics as weight of the note
    max_amp = 0
    num_harmonic = 1
    harmonic = num_harmonic * fund
    top_freq = (len(F)-1) * freqs[1] 
    while harmonic <= top_freq:
        # Compute the indexes of the nearest two harmonics of window's fund. to harmonic
        h1,h2 = indexes(freqs,0,len(freqs)-1,harmonic) 
        if h2 < 0:
            max_amp = max([max_amp,F[h1]])
        else:
            # Weighted mean of F[h1] and F[h2] by distance of freqs[h1] and freqs[h2] to the harmonic
            if freqs[h1] != 0:
                max_amp = max([max_amp,(F[h1]*np.log2(freqs[h2]/harmonic) + F[h2]*np.log2(harmonic/freqs[h1])) / np.log2(freqs[h2]/freqs[h1])])

        num_harmonic += 1
        harmonic = num_harmonic * fund
        
    return max_amp
    
def detect_note(fft):
    freqs = np.fft.rfftfreq(WINDOW_SIZE_SAMPLES, 1/SAMPLE_RATE) # The array of frequencies to evaluate in the fft
    F = np.abs(fft.real) # Evaluations of those frequencies
    
    peaks = [number_to_freq(round(freq_to_number(i))) for i in detect_peaks(freqs,F)] # Round to equal temperament
    if not len(peaks):
        return ('S',0)
    peaks = remove_duplicates(peaks) # Remove duplicates
     
    candidates = find_candidates(peaks)
    nharmonics = count_harmonics(peaks,candidates,freqs[-1]) 

    pred_fund = candidates[np.argmax(nharmonics)]
    
    return (note_name(int(round(freq_to_number(pred_fund)))),max_amplitude(pred_fund,freqs,F))

def correct_notes_iteration(notes,weights,nl,nr): # Correct each note according to its nl previous and nr following ones' weights
    cnotes = notes.copy() # Necessary for keeping the first and last notes
    cweights = weights.copy() # Necessary for keeping the first and last weights
    
    silence_threshold = SILENCE_THRESHOLD * max(cweights)
    n = max(nl,nr)
    w = [1] # New weights for the window, based on proximity to the note to be corrected
    for k in range(1,n+1):
        w.append(sum(w))
        
    for i in range(0,len(notes)-nr):
        if cnotes[i] == 'S': # Avoid correcting silence
            continue
        if cweights[i] <= silence_threshold: # Correct as silence and keep the weight
            cnotes[i] = 'S'
            continue
            
        if i in range(0,nl) or i in range(len(notes)-nr,len(notes)): # Skip if cannot be corrected for being out of the range
            continue
        
        nsubset = []
        wsubset = []
        nsums = []
        for j in range(i-nl,i+nr+1):
            if notes[j] not in nsubset:
                nsubset.append(notes[j])
                if j <= i:
                    wsubset.append(w[j-(i-n)] * weights[j])
                    nsums.append(w[j-(i-n)])
                else:
                    wsubset.append(w[i+n-j] * weights[j] / 2) # Little penalization to future notes
                    nsums.append(w[i+n-j] / 2)
            else:
                if j <= i:
                    wsubset[nsubset.index(notes[j])] += w[j-(i-n)] * weights[j]
                    nsums[nsubset.index(notes[j])] += w[j-(i-n)]
                else:
                    wsubset[nsubset.index(notes[j])] += w[i+n-j] * weights[j] / 2
                    nsums[nsubset.index(notes[j])] += w[i+n-j] / 2
                  
        index = len(wsubset) - wsubset[::-1].index(max(wsubset)) - 1 # Index of last maximum of wsubset
        cnotes[i] = nsubset[index]
        cweights[i] = wsubset[index] / nsums[index]
        
    return cnotes,cweights

def correct_notes(notes,weights,nl,nr): # Correct the notes
    count = 0
    
    notes_before = notes.copy()
    notes_after,cweights = correct_notes_iteration(notes,weights,nl,nr)
    while not np.array_equal(notes_before,notes_after) and count < MAX_CORRECTIONS:
        notes_before = notes_after.copy()
        count += 1
        notes_after,cweights = correct_notes_iteration(notes_before,cweights,nl,nr)
    
    return notes_before

def notes_per_window():
    hanning = 0.5 * (1 - np.cos(np.linspace(0,2*np.pi,WINDOW_SIZE_SAMPLES,False)))  # The hanning window function

    notes = []
    weights = []
    window_number = 0
    last_window = False
    while not(last_window):
        last_window, window = extract_window(SIGNAL,window_number)
        window_number += 1
        fft = np.fft.rfft(autocorrelation(window * hanning))
        note,weight = detect_note(fft)
        notes.append(note)
        weights.append(weight)
       
    return correct_notes(notes,weights,N_NOTES_CORRECTION_L,N_NOTES_CORRECTION_R)

# Computing the durations

def durations(n_prev,d_prev): # Compute the arrays of notes and durations (in seconds) from previous array or not
    notes = []
    durations = []
    
    cn = n_prev[0] # Current note
    cnd = 0 # Current note's duration
    if not d_prev:
        for n in n_prev:
            if n == cn:
                cnd += WINDOW_SIZE_SECS - OVERLAPPING_SECS
            else:
                notes.append(cn)
                durations.append(cnd)
                cn = n
                cnd = WINDOW_SIZE_SECS - OVERLAPPING_SECS
        notes.append(cn)
        durations.append(cnd + OVERLAPPING_SECS)
    else:
        for i in range(0,len(n_prev)):
            if n_prev[i] == cn:
                cnd += d_prev[i]
            else:
                notes.append(cn)
                durations.append(cnd)
                cn = n_prev[i]
                cnd = d_prev[i]
        notes.append(cn)
        durations.append(cnd)
      
    return notes,durations

def to_quarters(durations,subdivision): # Compute the array of durations (in quarters)
    return np.round(subdivision * np.array(durations) / QUARTER_SECS) / subdivision

def remove_zeros(n,d,q,remove_rests): 
    notes = []
    durations = []
    quarters = []
    for i in range(0,len(n)):
        if q[i] > 0 or (n[i] == 'S' and not remove_rests):
            notes.append(n[i])
            durations.append(d[i])
            quarters.append(q[i])
            
    return notes,durations,quarters

def final_notes_durations(npw,subdivision): # Compute the final arrays of notes and durations (in seconds and quarters)
    n_before = npw.copy()
    d = []
    q = []
    
    n_after,d = durations(n_before,d)
    q = to_quarters(d,subdivision)
    n_after,d,q = remove_zeros(n_after,d,q,False)
    while not np.array_equal(n_before,n_after):
        n_before = n_after.copy()
        n_after,d = durations(n_before,d)
        q = to_quarters(d,subdivision)
        n_after,d,q = remove_zeros(n_after,d,q,False)
    
    return remove_zeros(n_before,d,q,True) # To considering little rests as separators between equal notes

# Conversion

def music_sheet(notes,quarters):
    score = stream.Score() # Score
    part = stream.Part() # Staff
    voice = stream.Voice() # Voice
    
    voice.append(meter.TimeSignature(str(SIGNATURE_NUM) + '/' + str(SIGNATURE_DEN))) # Define time signature
    
    first_note = False # We allow clef changes after insertion of the fist note (not rest)
    for i in range(0,len(notes)):
        if notes[i] == 'S':
            voice.append(note.Rest(quarters[i]))
            continue
        n = note.Note(notes[i])
        n.duration = duration.Duration(quarters[i])
        
        cclef = None
        if voice.getElementsByClass('Clef'): # Get current clef
            cclef = voice.getElementsByClass('Clef')[len(voice.getElementsByClass('Clef'))-1]
        else:
            cclef = clef.bestClef(voice)

        if first_note and n.octave < 4 and cclef == clef.TrebleClef(): # Condition to change to F clef
            voice.append(clef.BassClef())
        if first_note and n.octave > 3 and cclef == clef.BassClef(): # Condition to change to G clef
            voice.append(clef.TrebleClef())
        
        voice.append(n)
        first_note = True
        
        
    quarters_in_measure = 4 * SIGNATURE_NUM / SIGNATURE_DEN
    last_silence_quarters = quarters_in_measure - (sum(quarters) % quarters_in_measure)
    voice.append(note.Rest(last_silence_quarters)) # Last silence duration to end last measure
    
    part.append([voice])
    score.insert(0,part)
    
    score.insert(0, metadata.Metadata())
    score.metadata.title = TITLE
    score.metadata.composer = COMPOSER
    
    score.write('musicxml.pdf',fp=FILENAME+'.pdf')
    score.write('midi',fp=FILENAME+'.midi')
    
# Conversion

def convert(): 
    global WINDOW_SIZE_SECS
    global SAMPLE_RATE
    global SIGNAL
    global WINDOW_SIZE_SAMPLES
    global OVERLAPPING_SAMPLES
    global AUDIO_SIZE_SECS
    global QUARTER_SECS
    
    SAMPLE_RATE, data = wavfile.read(os.path.join(PATH,AUDIO_FILE))      # Get sample rate (samples per second) and signal data
    SIGNAL = data if data.ndim == 1 else data.T[0]                       # Get the first channel
    
    x = []
    y = []
    for f in [110.0,120.0,130.0,140.0,150.0]:
        WINDOW_SIZE_SECS = 1.0 / f
        OVERLAPPING_SECS = WINDOW_SIZE_SECS / 2.0
        WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_SECS)            # Size of the fft window in samples
        OVERLAPPING_SAMPLES = int(SAMPLE_RATE * OVERLAPPING_SECS)            # Size of overlapping in samples
        AUDIO_SIZE_SECS = len(SIGNAL) / SAMPLE_RATE                          # Size of the audio file in seconds

        QUARTER_SECS = 60.0 / QUARTER_PPM                                    # Seconds of a quarter
                    
        start = timer()
        npw = notes_per_window()
        notes,_,quarters = final_notes_durations(npw,QUANTIZATION_FACTOR)
        music_sheet(notes,quarters)
        end = timer()
        print("Windows fund freq: " + str(1.0/WINDOW_SIZE_SECS) + ". Elapsed time (s): " + str(end-start))
        x.append(1.0/WINDOW_SIZE_SECS)
        y.append(end-start)
    print(x)
    print(y)    
    
    
# Input reading and settings

def submit(): # Validation of inputs and converter's execution
    try:
        global AUDIO_FILE
        global WINDOW_SIZE_SECS
        global OVERLAPPING_SECS
        global SILENCE_THRESHOLD
        global INTENSITY_THRESHOLD
        global MAX_REL_PEAKS
        global MAX_KM_ITERATIONS
        global N_NOTES_CORRECTION_L
        global N_NOTES_CORRECTION_R
        global MAX_CORRECTIONS
        global QUARTER_PPM
        global QUANTIZATION_FACTOR
        global SIGNATURE_NUM
        global SIGNATURE_DEN
        global TITLE
        global COMPOSER
        global FILENAME
        
        AUDIO_FILE = entry_audio_file.get()
        WINDOW_SIZE_SECS = float(entry_window_size_secs.get())
        OVERLAPPING_SECS = float(entry_overlapping_secs.get())
        SILENCE_THRESHOLD = float(entry_silence_threshold.get())
        INTENSITY_THRESHOLD = float(entry_intensity_threshold.get())
        MAX_REL_PEAKS = int(entry_max_rel_peaks.get())
        MAX_KM_ITERATIONS = int(entry_max_km_iterations.get())
        N_NOTES_CORRECTION_L = int(entry_n_notes_correction_l.get())
        N_NOTES_CORRECTION_R = int(entry_n_notes_correction_r.get())
        MAX_CORRECTIONS = int(entry_max_corrections.get())
        QUARTER_PPM = int(entry_quarter_ppm.get())
        QUANTIZATION_FACTOR = int(entry_quantization_factor.get())
        SIGNATURE_NUM = int(entry_signature_num.get())
        SIGNATURE_DEN = int(entry_signature_den.get())
        TITLE = entry_title.get()
        COMPOSER = entry_composer.get()
        FILENAME = entry_filename.get()

        # Validation checks
        if not os.path.isfile(AUDIO_FILE):
            raise ValueError("The audio file must exist in the working directory.")
        if WINDOW_SIZE_SECS <= 0:
            raise ValueError("WINDOW_SIZE_SECS must be greater than 0.")
        if OVERLAPPING_SECS < 0:
            raise ValueError("OVERLAPPING_SECS must be greater than or equal to 0.")
        if WINDOW_SIZE_SECS <= OVERLAPPING_SECS:
            raise ValueError("WINDOW_SIZE_SECS must be greater than OVERLAPPING_SECS.")
        if not (0 <= SILENCE_THRESHOLD <= 1):
            raise ValueError("SILENCE_THRESHOLD must be between 0 and 1 (inclusive).")
        if INTENSITY_THRESHOLD <= 0:
            raise ValueError("INTENSITY_THRESHOLD must be greater than 0.")
        if MAX_REL_PEAKS <= 0:
            raise ValueError("MAX_REL_PEAKS must be greater than 0.")
        if MAX_KM_ITERATIONS <= 0:
            raise ValueError("MAX_KM_ITERATIONS must be greater than 0.")
        if N_NOTES_CORRECTION_L < 0:
            raise ValueError("N_NOTES_CORRECTION_L must be greater than or equal to 0.")
        if N_NOTES_CORRECTION_R < 0:
            raise ValueError("N_NOTES_CORRECTION_R must be greater than or equal to 0.")
        if MAX_CORRECTIONS < 0:
            raise ValueError("MAX_CORRECTIONS must be greater than or equal to 0.")
        if QUARTER_PPM <= 0:
            raise ValueError("QUARTER_PPM must be greater than 0.")
        if QUANTIZATION_FACTOR not in {1, 2, 4, 8}:
            raise ValueError("QUANTIZATION_FACTOR must be one of {1, 2, 4, 8}.")
        if SIGNATURE_NUM <= 0:
            raise ValueError("SIGNATURE_NUM must be greater than 0.")
        if SIGNATURE_DEN not in {1, 2, 4, 8, 16}:
            raise ValueError("SIGNATURE_DEN must be one of {1, 2, 4, 8, 16}.")
        if not FILENAME:
            raise ValueError("FILENAME must not be empty.")

        messagebox.showinfo("Success","Parameters received correctly.")
        
        convert()
    except ValueError as e:
        messagebox.showerror("Error",f"Invalid input:{e}")
        
def toggle_advanced_params(entries,advanced_entries,labels,toggle_button,root):
    for key,(_,entry) in entries.items():
        if key in advanced_entries:
            if entry.winfo_ismapped(): # If visible, hide
                entry.grid_remove()
                labels[key].grid_remove()
                toggle_button.config(text="Show Advanced Settings")
                root.geometry(WDIMENSIONS['little'])
            else: # Si hidden, show
                entry.grid()
                labels[key].grid()
                toggle_button.config(text="Hide Advanced Settings")
                root.geometry(WDIMENSIONS['large'])
        
# Main

def main():
    # Create the main window
    custom_font = ('Courier',9)
    root = tk.Tk()
    root.title("W2MC")
    
    root.geometry(WDIMENSIONS['little']) # Set fixed size for the window
    root.resizable(False, False)
    
    icon = ImageTk.PhotoImage(file='icon.ico')   
    root.tk.call('wm','iconphoto',root._w,icon) # Program's icon
    
    root.config(bg='black')
    
    title_frame = tk.Frame(root,bg='black') # Container for a title
    title_frame.grid(row=0,column=0,columnspan=3,pady=30)
    title_label = tk.Label(title_frame,text="WAV2MIDIConverter",font=('Courier',16),bg='black',fg='gold')
    title_label.pack()
    
    entry_frame = tk.Frame(root,bg='black') # Container for entries and labels
    
    entries = { # Create the widgets
        "AUDIO_FILE": ("Audio file's name:",tk.Entry(entry_frame)),
        "WINDOW_SIZE_SECS": ("Size of the fft window (s):",tk.Entry(entry_frame)),
        "OVERLAPPING_SECS": ("Window's overlapping (s):",tk.Entry(entry_frame)),
        "SILENCE_THRESHOLD": ("Silence threshold:",tk.Entry(entry_frame)),
        "INTENSITY_THRESHOLD": ("Intensity threshold:",tk.Entry(entry_frame)),
        "MAX_REL_PEAKS": ("Maximum number of relevant peaks:",tk.Entry(entry_frame)),
        "MAX_KM_ITERATIONS": ("Maximum number of K-Means iterations:",tk.Entry(entry_frame)),
        "N_NOTES_CORRECTION_L": ("Number of notes to the left for correction:",tk.Entry(entry_frame)),
        "N_NOTES_CORRECTION_R": ("Number of notes to the right for correction:",tk.Entry(entry_frame)),
        "MAX_CORRECTIONS": ("Maximum number of corrections:",tk.Entry(entry_frame)),
        "QUARTER_PPM": ("Tempo in terms of a quarter:",tk.Entry(entry_frame)),
        "QUANTIZATION_FACTOR": ("Quantization factor:",tk.Entry(entry_frame)),
        "SIGNATURE_NUM": ("Signature numerator:",tk.Entry(entry_frame)),
        "SIGNATURE_DEN": ("Signature denominator:",tk.Entry(entry_frame)),
        "TITLE": ("Title of the piece:",tk.Entry(entry_frame)),
        "COMPOSER": ("Composer of the piece:",tk.Entry(entry_frame)),
        "FILENAME": ("PDF file's name:",tk.Entry(entry_frame))
    }
    
    advanced_entries = [ # Define the advanced settings
        "WINDOW_SIZE_SECS",
        "OVERLAPPING_SECS",
        "SILENCE_THRESHOLD",
        "INTENSITY_THRESHOLD",
        "MAX_REL_PEAKS",
        "MAX_KM_ITERATIONS",
        "N_NOTES_CORRECTION_L",
        "N_NOTES_CORRECTION_R",
        "MAX_CORRECTIONS"
    ]
    
    entry_frame.grid(row=1,column=0,rowspan=len(entries),columnspan=2,padx=10,pady=5,sticky="nsew")
    
    # Set default values and place labels and entries in the window
    labels = {}
    for i,(key,(label_text,entry)) in enumerate(entries.items()):
        labels[key] = tk.Label(entry_frame,text=label_text,font=custom_font,bg='black',fg='gold')
        labels[key].grid(row=i,column=0,padx=10,pady=5,sticky="e")
        entry.config(font=custom_font,bg='gold')
        entry.grid(row=i,column=1,padx=10,pady=5)
        entry.insert(0,DEFAULTS[key])
        globals()[f"entry_{key.lower()}"] = entry  # Create dynamic entry variables
        if key in advanced_entries:
            entry.grid_remove()
            labels[key].grid_remove()
            
    side_image = Image.open("clef.png") # Side image
    side_image = ImageTk.PhotoImage(side_image)
    side_label = tk.Label(root,image=side_image,bg='black')
    side_label.grid(row=1,column=2,rowspan=len(entries),padx=0,pady=0)
        
    # Create a button to showing/hiding the advanced settings
    toggle_button = tk.Button(entry_frame,text="Show Advanced Settings",font=custom_font,bg='gold',command=lambda: toggle_advanced_params(entries,advanced_entries,labels,toggle_button,root))
    toggle_button.grid(row=len(entries),columnspan=2,pady=10)

    button_submit = tk.Button(entry_frame,text="Submit",font=custom_font,bg='gold',command=submit)
    button_submit.grid(row=len(entries)+1,columnspan=2,pady=10)

    root.mainloop() # Run the main application loop

# -------------------------Execution------------------------

if __name__ == "__main__":
    main()