# 2MANYFILTERS
# Javier Granados López
# TFG 2023/2024

# --------------------------Imports--------------------------
import sys
from scipy.io import wavfile
import os
import numpy as np

import tkinter as tk
from tkinter import messagebox
from PIL import Image,ImageTk

# --------------------------Inputs--------------------------
AUDIO_FILE = 'name'                                                  # Audio file's name
WINDOW_SIZE_SECS = 0.2                                               # Size of the fft window in seconds
FILTER_TYPE = 'copy'                                                 # Filter type in [LPF,HPF,BPF,BSF]; other does nothing
THRESHOLD1 = 20                                                      # First threshold for the filter in Hz
THRESHOLD2 = 20000                                                   # Second threshold for the filter (if necessary) in Hz
FILENAME = 'filename'                                                # The WAV file's name

# ---------------------Global variables---------------------
WDIMENSIONS = '1050x500'
DEFAULTS = {                                                         # Default values 
    "AUDIO_FILE" : AUDIO_FILE,
    "WINDOW_SIZE_SECS": WINDOW_SIZE_SECS,
    "FILTER_TYPE": FILTER_TYPE,
    "THRESHOLD1": THRESHOLD1,
    "THRESHOLD2": THRESHOLD2,
    "FILENAME": FILENAME                                           
}
PATH = os.path.dirname(os.path.realpath(sys.argv[0]))                # Path of project's directory

OVERLAPPING_SECS = None                                              # Window's overlapping in seconds
SAMPLE_RATE = None                                                   # Sample rate (samples per second)
SIGNAL = None                                                        # Signal data
WINDOW_SIZE_SAMPLES = None                                           # Size of the fft window in samples
OVERLAPPING_SAMPLES = None                                           # Size of overlapping in samples
AUDIO_SIZE_SECS = None                                               # Size of the audio file in seconds
HANNING = None                                                       # The hanning window function

# -------------------------Functions-------------------------
# Filtering

def extract_window(audio, window_number):                                   # Returns samples of window number <window-number> and true or false whether it's the last window 
    begin = window_number * (WINDOW_SIZE_SAMPLES - OVERLAPPING_SAMPLES)
    end = begin + WINDOW_SIZE_SAMPLES
    
    if end < len(SIGNAL): # Commonly
        return False, audio[begin:end]
    else: # The window surpasses the audio data => Complete last elements of the window with zeros
        return True, np.concatenate([audio[begin:len(SIGNAL)-1],np.zeros(end-len(SIGNAL)+1,dtype=float)])
    
def analysis(window): # Compute the FFT's module curve and return x and y axes in a tuple
    freqs = np.fft.rfftfreq(WINDOW_SIZE_SAMPLES, 1/SAMPLE_RATE) # The array of frequencies to evaluate in the fft
    fft = np.fft.rfft(window) # Evaluations of those frequencies

    return (freqs,fft)

def filter_window(fft): # Filter the FFT's module function according to the selected filter and thresholds
    if FILTER_TYPE == 'LPF':
        filtered = [fft[1][i] if fft[0][i] <= THRESHOLD1 else 0 for i in range(0,len(fft[0]))]
    elif FILTER_TYPE == 'HPF':
        filtered = [fft[1][i] if fft[0][i] >= THRESHOLD1 else 0 for i in range(0,len(fft[0]))]
    elif FILTER_TYPE == 'BPF':
        filtered = [fft[1][i] if fft[0][i] >= THRESHOLD1 and fft[0][i] <= THRESHOLD2 else 0 for i in range(0,len(fft[0]))]
    elif FILTER_TYPE == 'BSF':
        filtered = [fft[1][i] if fft[0][i] <= THRESHOLD1 or fft[0][i] >= THRESHOLD2 else 0 for i in range(0,len(fft[0]))]
    else:
        filtered = fft[1].copy()

    return filtered

def synthesis(ffts): # Synthesize the new signal via the list of windows considering overlapping
    new_signal = (np.real(np.fft.irfft(ffts[0])) * HANNING).astype(SIGNAL.dtype)
    for i in range(1,len(ffts)):
        window = (np.real(np.fft.irfft(ffts[i])) * HANNING).astype(SIGNAL.dtype)
        for j in range(0,OVERLAPPING_SAMPLES):
            new_signal[-OVERLAPPING_SAMPLES+j] = new_signal[-OVERLAPPING_SAMPLES+j] + window[j]
        new_signal = np.append(new_signal,window[OVERLAPPING_SAMPLES:])
    
    return new_signal

def filtering():
    filtered_windows = []
    window_number = 0
    last_window = False
    while not(last_window):
        last_window, window = extract_window(SIGNAL,window_number)
        window_number += 1
        filtered_windows.append(filter_window(analysis(window)))
    
    filtered_signal = synthesis(filtered_windows)
    wavfile.write(os.path.join(PATH,FILENAME+".wav"),SAMPLE_RATE,filtered_signal)
    
# Input reading and settings

def submit(): # Validation of inputs and converter's execution
    try:
        global AUDIO_FILE
        global WINDOW_SIZE_SECS
        global FILTER_TYPE
        global THRESHOLD1
        global THRESHOLD2
        global FILENAME
        
        AUDIO_FILE = entry_audio_file.get()
        WINDOW_SIZE_SECS = float(entry_window_size_secs.get())
        FILTER_TYPE = entry_filter_type.get()
        THRESHOLD1 = float(entry_threshold1.get())
        THRESHOLD2 = float(entry_threshold2.get())
        FILENAME = entry_filename.get()

        # Validation checks
        if not os.path.isfile(AUDIO_FILE):
            raise ValueError("The audio file must exist in the working directory.")
        if WINDOW_SIZE_SECS <= 0:
            raise ValueError("WINDOW_SIZE_SECS must be greater than 0.")
        if THRESHOLD1 <= 0:
            raise ValueError("THRESHOLD1 must be greater than 0.")
        if THRESHOLD2 <= 0:
            raise ValueError("THRESHOLD2 must be greater than 0.")
        if THRESHOLD2 < THRESHOLD1:
            raise ValueError("THRESHOLD1 must be greater than or equal to THRESHOLD2.")
        if not FILENAME:
            raise ValueError("FILENAME must not be empty.")

        messagebox.showinfo("Success","Parameters received correctly.")
        
        global OVERLAPPING_SECS
        global HANNING
        global SAMPLE_RATE
        global SIGNAL
        global WINDOW_SIZE_SAMPLES
        global OVERLAPPING_SAMPLES
        global AUDIO_SIZE_SECS
        
        OVERLAPPING_SECS = WINDOW_SIZE_SECS / 2 # Mandatory because of the fixed hanning function
        SAMPLE_RATE, data = wavfile.read(os.path.join(PATH,AUDIO_FILE))      # Get sample rate (samples per second) and signal data
        SIGNAL = data if data.ndim == 1 else data.T[0]                       # Get the first channel
        WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_SECS)            # Size of the fft window in samples
        OVERLAPPING_SAMPLES = int(SAMPLE_RATE * OVERLAPPING_SECS)            # Size of overlapping in samples
        AUDIO_SIZE_SECS = len(SIGNAL) / SAMPLE_RATE                          # Size of the audio file in seconds
        HANNING = 0.5 * (1 - np.cos(np.linspace(0,2*np.pi,WINDOW_SIZE_SAMPLES,False)))
        
        # ---------------------Files' statistics---------------------
        messagebox.showinfo("File's info","Sample rate: " + str(SAMPLE_RATE) + " Hz\n" +                                                       
        "Window size: " + str(WINDOW_SIZE_SECS) + " s = " + str(WINDOW_SIZE_SAMPLES) + " samples\n" +
        "Overlapping: " + str(OVERLAPPING_SECS) + " s = " + str(OVERLAPPING_SAMPLES) + " samples\n" +
        "Audio length: " + str(AUDIO_SIZE_SECS) + " s\n")
        
        filtering()
    except ValueError as e:
        messagebox.showerror("Error",f"Invalid input:{e}")
        
def toggle_advanced_params(entries,advanced_entries,labels,toggle_button,root):
    for key,(_,entry) in entries.items():
        if key in advanced_entries:
            if entry.winfo_ismapped(): # If visible, hide
                entry.grid_remove()
                labels[key].grid_remove()
                toggle_button.config(text="Show Advanced Settings")
                root.geometry(WDIMENSIONS)
            else: # Si hidden, show
                entry.grid()
                labels[key].grid()
                toggle_button.config(text="Hide Advanced Settings")
                root.geometry(WDIMENSIONS)
                
# Auxiliary

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
        
# Main

def main():
    # Create the main window
    custom_font = ('Courier',9)
    root = tk.Tk()
    root.title('2MF')
    
    root.geometry(WDIMENSIONS) # Set fixed size for the window
    root.resizable(False,False)
    
    icon = ImageTk.PhotoImage(file=resource_path('icon.ico'))   
    root.tk.call('wm','iconphoto',root._w,icon) # Program's icon

    root.config(bg='black')
    
    title_frame = tk.Frame(root,bg='black') # Container for a title
    title_frame.grid(row=0,column=0,columnspan=3,pady=30)
    title_label = tk.Label(title_frame,text="2MANYFILTERS",font=('Courier',16),bg='black',fg='gold')
    title_label.pack()
    
    entry_frame = tk.Frame(root,bg='black') # Container for entries and labels
    
    entries = { # Create the widgets
        "AUDIO_FILE": ("Audio file's name:",tk.Entry(entry_frame)),
        "WINDOW_SIZE_SECS": ("Size of the fft window (s):",tk.Entry(entry_frame)),
        "FILTER_TYPE": ("Filter type in {LPF,HPF,BPF,BSF} (other does nothing):",tk.Entry(entry_frame)),
        "THRESHOLD1": ("First threshold for the filter (Hz):",tk.Entry(entry_frame)),
        "THRESHOLD2": ("Second threshold for the filter (if necessary) in (Hz):",tk.Entry(entry_frame)),
        "FILENAME": ("WAV file's name:",tk.Entry(entry_frame))
    }
    
    advanced_entries = ["WINDOW_SIZE_SECS"] # Define the advanced settings
    
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
            
    side_image = Image.open(resource_path('filter.png')) # Side image
    side_image = ImageTk.PhotoImage(side_image)
    side_label = tk.Label(root,image=side_image,bg='black')
    side_label.grid(row=1,column=2,rowspan=len(entries),padx=40,pady=0)
        
    # Create a button to showing/hiding the advanced settings
    toggle_button = tk.Button(entry_frame,text="Show Advanced Settings",font=custom_font,bg='gold',command=lambda: toggle_advanced_params(entries,advanced_entries,labels,toggle_button,root))
    toggle_button.grid(row=len(entries),columnspan=2,pady=10)

    button_submit = tk.Button(entry_frame,text="Submit",font=custom_font,bg='gold',command=submit)
    button_submit.grid(row=len(entries)+1,columnspan=2,pady=10)

    root.mainloop() # Run the main application loop

# -------------------------Execution------------------------

if __name__ == "__main__":
    main()