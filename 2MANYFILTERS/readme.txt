Para convertir los ficheros .py en archivos ejecutables .exe:
- Instalar pyinstaller
- (Excluir el directorio de trabajo del control de Windows Defender porque identifica los nuevos archivos como troyanos)
- Ejecutar este modelo de comando en el directorio: pyinstaller --onefile --windowed --add-data "<archivo.png>:." <archivo.py>
Para 2MANYFILTERS: pyinstaller --onefile --windowed --add-data "filter.png:." --add-data "icon.ico:." 2mf.py
Para WAV2MIDIConverter: pyinstaller --onefile --windowed --add-data "clef.png:." --add-data "icon.ico:." w2mc.py