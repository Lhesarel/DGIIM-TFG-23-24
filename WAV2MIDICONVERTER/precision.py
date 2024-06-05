# Computing accuracy of the different detection methods
# Javier Granados López
# TFG 2023/2024

def precision(arr1,arr2):
    matches = 0

    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            matches += 1
    percentage = 100 * matches / len(arr1)
    
    return percentage

solution = ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4']
predictions = {
    "1" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C5', 'C5', 'C5', 'C4', 'C5', 'G5', 'D4', 'D4', 'D5', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D5', 'D4', 'F4', 'E6', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'B1', 'D#0', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F5', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'G#4', 'D6', 'G4', 'G5', 'G4', 'G5', 'G4', 'G5', 'G4', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A5', 'D#0', 'D#0', 'D#0', 'A1', 'D#0', 'D#0', 'B4', 'B4', 'B4', 'B4', 'B5', 'B4', 'B4', 'B4', 'B4', 'F#6', 'B4', 'B4', 'B5', 'B4', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'B4', 'C5', 'C5', 'C5', 'C5', 'C6', 'C5', 'C6', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B5', 'B4', 'A1', 'B4', 'B4', 'A1', 'D#0', 'D#0', 'A4', 'A4', 'A4', 'A4', 'A5', 'A5', 'A5', 'E6', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'B1', 'B1', 'B1', 'G#4', 'G4', 'G5', 'G4', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G4', 'G5', 'B1', 'G5', 'G4', 'G5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F5', 'F5', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E5', 'E4', 'E6', 'E5', 'E6', 'E5', 'E6', 'E6', 'E5', 'E6', 'B5', 'A1', 'D#0', 'D#0', 'D#0', 'D#0', 'A1', 'E6', 'D4', 'D4', 'D4', 'F#6', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D5', 'D4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C5', 'C4', 'C5', 'C4', 'C4', 'C5', 'A1', 'G5', 'C4', 'G5', 'G5', 'D#0', 'C5', 'D#0', 'G5', 'C5', 'D#0', 'D#0'],
    "2" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D#0', 'D#0', 'D#0', 'D4', 'C#4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D#0', 'D4', 'D4', 'D#4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E5', 'E5', 'A1', 'E5', 'A1', 'E5', 'D#0', 'D#0', 'D#0', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F#4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'A1', 'G5', 'D#0', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A1', 'B4', 'B4', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C6', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'A1', 'D#0', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A1', 'B4', 'B4', 'B4', 'B4', 'A1', 'B1', 'B4', 'D#0', 'D#0', 'D#0', 'G#4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A5', 'D#0', 'B1', 'A4', 'A4', 'A4', 'A4', 'A4', 'A1', 'B1', 'F#1', 'D#0', 'F4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G5', 'G1', 'G4', 'G4', 'G5', 'B1', 'B1', 'B1', 'C2', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'D#4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E5', 'D#0', 'B5', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D4', 'D4', 'D4', 'F#6', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D#0', 'D4', 'D4', 'D#0', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'D#0', 'C5', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0', 'D#0'],
    "3" : ['C4', 'C4', 'C4', 'C4', 'C5', 'E6', 'C5', 'C5', 'G5', 'C5', 'C5', 'A#6', 'C5', 'A#6', 'C5', 'G5', 'A#6', 'D4', 'F#6', 'D4', 'D4', 'F#6', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D5', 'D4', 'E5', 'E6', 'E5', 'E6', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'B6', 'E5', 'B6', 'E7', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F5', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'D6', 'D7', 'G4', 'G5', 'D6', 'G5', 'D7', 'G5', 'D7', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'D7', 'G5', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A5', 'A4', 'A4', 'A4', 'A4', 'B4', 'B4', 'B4', 'B4', 'B5', 'B4', 'B4', 'B6', 'B4', 'F#6', 'B4', 'B4', 'B5', 'B4', 'B4', 'B5', 'B4', 'B4', 'B5', 'C6', 'G6', 'C5', 'G6', 'C5', 'G6', 'C5', 'G6', 'C5', 'G6', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'F#7', 'B4', 'B4', 'B6', 'B4', 'B4', 'B4', 'B4', 'B5', 'B4', 'B4', 'F#6', 'B4', 'B4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A5', 'A5', 'E6', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A5', 'D6', 'G4', 'G5', 'D7', 'G5', 'D7', 'G5', 'G5', 'G5', 'G5', 'D7', 'G5', 'G4', 'G5', 'F7', 'G5', 'F7', 'G5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F5', 'F5', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E4', 'E5', 'E6', 'E6', 'E5', 'E6', 'E5', 'E6', 'E6', 'B6', 'E6', 'E6', 'B6', 'E6', 'B6', 'E7', 'B6', 'E6', 'E6', 'C7', 'D4', 'D4', 'F#6', 'D4', 'D4', 'D6', 'D4', 'D4', 'D4', 'D4', 'D6', 'D4', 'D4', 'D5', 'D4', 'D4', 'D5', 'D4', 'C5', 'C4', 'C5', 'C4', 'C5', 'C5', 'C4', 'C4', 'C5', 'E6', 'C5', 'A#6', 'C5', 'A#6', 'C6', 'G5', 'A#6', 'C5', 'E6', 'D7', 'G5', 'E6', 'G5', 'E6', 'E6', 'C5'],
    "4" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C3', 'D#-1', 'D3', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'B0', 'D#0', 'D#-1', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'G#4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'D#0', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'D#-1', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'D#-1', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'D#0', 'D#-1', 'D#-1', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'B0', 'B0', 'B0', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G3', 'G5', 'D#0', 'G4', 'G4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E4', 'E5', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'D#-1', 'D#-1', 'D#-1', 'D#0', 'D#-1', 'B1', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D#-1', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D#0', 'C4', 'D#-1', 'C4', 'C4', 'D#-1', 'D#-1', 'D#-1', 'D#0', 'D#0', 'D#0', 'D#-1'],
    "5" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'A#0', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D4', 'D4', 'C1', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E5', 'E4', 'A0', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'B0', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A4', 'A4', 'A4', 'A4', 'A4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B5', 'B4', 'B4', 'B4', 'B4', 'B4', 'A0', 'A0', 'D1', 'C5', 'C5', 'C5', 'C5', 'C5', 'C6', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A1', 'B4', 'B4', 'A#0', 'B0', 'A4', 'A4', 'A4', 'A4', 'A5', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A0', 'C1', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'C1', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F5', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'A0', 'F1', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'A0', 'A1', 'A0', 'D4', 'D4', 'D4', 'F#6', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D5', 'D4', 'D4', 'D5', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'A#0', 'F1'],
    "6" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D4', 'D4', 'D4', 'D4', 'D4', 'G2', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'G2', 'D4', 'D4', 'D3', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'A#2', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'C#3', 'G4', 'G4', 'C4', 'G4', 'G4', 'G4', 'C4', 'G4', 'G4', 'G4', 'C4', 'G4', 'G4', 'C4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A0', 'A4', 'A4', 'E3', 'E3', 'B4', 'E3', 'B4', 'B4', 'B4', 'E3', 'E3', 'B4', 'B4', 'B4', 'E4', 'B4', 'B4', 'A0', 'A0', 'A0', 'A0', 'C5', 'C5', 'F3', 'F3', 'F3', 'C4', 'F3', 'F4', 'F3', 'C3', 'F3', 'F3', 'F3', 'F3', 'F3', 'F3', 'F3', 'F3', 'E3', 'E3', 'B4', 'E3', 'B4', 'E3', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A0', 'B4', 'B4', 'A0', 'B4', 'B4', 'G#3', 'D3', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A#0', 'A#0', 'B0', 'G#2', 'G4', 'C4', 'G4', 'G4', 'G4', 'C4', 'G4', 'G4', 'C4', 'G4', 'G4', 'G4', 'C4', 'A#0', 'G4', 'G4', 'C4', 'A#2', 'F4', 'A#2', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'A#2', 'A#2', 'F4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'A0', 'E4', 'E4', 'E4', 'A#0', 'A0', 'E4', 'D4', 'D4', 'D4', 'F#4', 'D4', 'D4', 'D4', 'G2', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D3', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'A0', 'C4', 'C4', 'C4', 'C4', 'A0', 'C4', 'C4', 'C4', 'C4', 'B0', 'F1'],
    "7" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E5', 'E5', 'E5', 'E5', 'E4', 'E4', 'E4', 'E4', 'E4', 'A0', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'A0', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G5', 'G5', 'G5', 'G5', 'G5', 'G4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A0', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A1', 'B4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A0', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G5', 'G5', 'G5', 'G5', 'G5', 'G5', 'G4', 'G4', 'G4', 'G4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'A0', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'E5', 'A1', 'A1', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'A#0', 'A0'],
    "8" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C1', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'D#1', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'D5', 'E5', 'E4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F1', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'A1', 'G5', 'C3', 'A1', 'G1', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A1', 'A1', 'A5', 'A5', 'A1', 'A1', 'A1', 'A4', 'A#0', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A1', 'B4', 'B4', 'A1', 'B5', 'A1', 'A1', 'A1', 'A#0', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C3', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'A1', 'B1', 'A#0', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B1', 'E2', 'B4', 'B1', 'A1', 'G#4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A5', 'A4', 'B1', 'A4', 'A4', 'A4', 'A4', 'A4', 'A1', 'B1', 'F#1', 'F#1', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'G5', 'G4', 'G4', 'G4', 'C2', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'D#1', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'Unknown', 'B1', 'A1', 'B1', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'B1', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'B0', 'F#1'],
    "9" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A1', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C2', 'C5', 'C5', 'C5', 'C5', 'C3', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B1', 'B4', 'B4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E5', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3'],
    "C" : ['C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A1', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'C5', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'B4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'A4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'G4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'F4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'E4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'D4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4']
}

for pred in predictions: 
    percentage = precision(predictions[pred],solution)
    print("Name: " + pred + ". Similarity percentaje: " + str(percentage))