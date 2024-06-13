<<<<<<< HEAD
Para convertir los ficheros .py en archivos ejecutables:
En Windows:
- Instalar pyinstaller
- (Excluir el directorio de trabajo del control de Windows Defender porque identifica los nuevos archivos como troyanos)
- Ejecutar este modelo de comando en el directorio: pyinstaller --onefile --windowed --add-data "<archivo.png>:." <archivo.py>
Para 2MANYFILTERS: pyinstaller --onefile --windowed --add-data "filter.png:." --add-data "icon.ico:." 2mf.py --hidden-import='PIL._tkinter_finder'
Para WAV2MIDIConverter: pyinstaller --onefile --windowed --add-data "clef.png:." --add-data "icon.ico:." w2mc.py --hidden-import='PIL._tkinter_finder'
En Linux
- Instalar pyinstaller:
- Instalar music21
Para 2MANYFILTERS: pyinstaller --onefile --windowed --add-data "filter.png:." --add-data "icon.ico:." 2mf.py --hidden-import='PIL._tkinter_finder'
Para WAV2MIDIConverter: pyinstaller --onefile --windowed --add-data "clef.png:." --add-data "icon.ico:." w2mc.py --hidden-import='PIL._tkinter_finder' --add-binary "<ruta/de/music21>:."
=======
Para convertir los ficheros .py en archivos ejecutables .exe:
- Instalar pyinstaller
- (Excluir el directorio de trabajo del control de Windows Defender porque identifica los nuevos archivos como troyanos)
- Ejecutar este modelo de comando en el directorio: pyinstaller --onefile --windowed --add-data "<archivo.png>:." <archivo.py>
Para 2MANYFILTERS: pyinstaller --onefile --windowed --add-data "filter.png:." --add-data "icon.ico:." 2mf.py
Para WAV2MIDIConverter: pyinstaller --onefile --windowed --add-data "clef.png:." --add-data "icon.ico:." w2mc.py
>>>>>>> 43d368c982914c90de4142e436173cdb42a3ef68
