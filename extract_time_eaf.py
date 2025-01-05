import pympi
import argparse
import os
import csv

parser = argparse.ArgumentParser(description='Process EAF files to extract gloss timings and transitions.')
parser.add_argument('--rootPath', type=str, default='./', help='Root path where EAF files are located')
parser.add_argument('--fileNameContains', type=str, default='ORACION', help='Substring that filenames must contain')
parser.add_argument('--fps', type=float, default=30.0, help='Frames per second of the videos')
args = parser.parse_args()

rootPath = args.rootPath
fileNameContains = args.fileNameContains
fps = args.fps

# Lista para almacenar los archivos EAF que cumplen con el criterio
listFile = []

# Recorrer recursivamente el directorio rootPath
for dirpath, dirnames, filenames in os.walk(rootPath):
    for filename in filenames:
        if filename.endswith('.eaf') and fileNameContains in filename:
            filePath = os.path.join(dirpath, filename)
            listFile.append(filePath)

print("Archivos a procesar:")
for file in listFile:
    print(file)

vocab = {}
gloss_list = []

for eafFilePath in listFile:
    # Open EAF file
    aEAFfile = pympi.Elan.Eaf(eafFilePath)
    eafFile = os.path.basename(eafFilePath)

    print(f"\nProcesando archivo: {eafFile}")
    # Reading gloss
    tier_names = ['GLOSA_IA']
    dictGloss = None
    for tier_name in tier_names:
        if tier_name in aEAFfile.tiers.keys():
            dictGloss = aEAFfile.tiers[tier_name]
            print(f"Tier de glosas encontrado: {tier_name}")
            break

    if dictGloss is None:
        print("No se encontró un tier de glosas en este archivo.")
        continue

    # Extraer las glosas y ordenarlas por tiempo de inicio
    annotations = []
    for annotation_id in dictGloss[0]:
        annotation = dictGloss[0][annotation_id]
        start_ts_id = annotation[0]
        end_ts_id = annotation[1]
        gloss_text = annotation[2]

        start_time = aEAFfile.timeslots[start_ts_id] / 1000  # Convert to seconds
        end_time = aEAFfile.timeslots[end_ts_id] / 1000      # Convert to seconds

        annotations.append({
            'gloss': gloss_text,
            'start_time': start_time,
            'end_time': end_time
        })

        # Update vocabulary
        vocab[gloss_text] = vocab.get(gloss_text, 0) + 1

    # Ordenar las glosas por tiempo de inicio
    annotations.sort(key=lambda x: x['start_time'])

    # Calcular los tiempos de transición y almacenar la información
    for i in range(len(annotations)):
        current_gloss = annotations[i]
        duration = current_gloss['end_time'] - current_gloss['start_time']
        frames = duration * fps

        gloss_list.append({
            'archivo': eafFile,
            'tipo': 'glosa',
            'glosa': current_gloss['gloss'],
            'inicio': current_gloss['start_time'],
            'fin': current_gloss['end_time'],
            'duracion': duration,
            'frames': int(round(frames))
        })

        # Imprimir información de la glosa
        print(f"Glosa: '{current_gloss['gloss']}' | Inicio: {current_gloss['start_time']}s | Fin: {current_gloss['end_time']}s | Duración: {duration}s | Frames: {int(round(frames))}")

        # Si no es la última glosa, calcular el tiempo de transición
        if i < len(annotations) - 1:
            next_gloss = annotations[i + 1]
            transition_start = current_gloss['end_time']
            transition_end = next_gloss['start_time']
            transition_duration = transition_end - transition_start

            if transition_duration > 0:
                transition_frames = transition_duration * fps
                gloss_list.append({
                    'archivo': eafFile,
                    'tipo': 'transicion',
                    'glosa': 'TRANSICION',
                    'inicio': transition_start,
                    'fin': transition_end,
                    'duracion': transition_duration,
                    'frames': int(round(transition_frames))
                })

                # Imprimir información de la transición
                print(f"Transición | Inicio: {transition_start}s | Fin: {transition_end}s | Duración: {transition_duration}s | Frames: {int(round(transition_frames))}")

print(f"\nTotal de glosas únicas encontradas: {len(vocab)}")

# Guardar los datos en un archivo CSV
csv_file = 'glosas_y_transiciones.csv'
csv_columns = ['archivo', 'tipo', 'glosa', 'inicio', 'fin', 'duracion', 'frames']

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in gloss_list:
            writer.writerow(data)
    print(f"\nLos resultados han sido guardados en el archivo {csv_file}")
except IOError:
    print("Ocurrió un error al escribir el archivo CSV")
