import os
import csv

def read_bbox_annotations(file_path, classes):
    annotations = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip().split()
            
            # Extrair informações das anotações
            class_label = int(line[0])
            class_name = classes[class_label]
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            
            annotations.append({
                'class_label': class_label,
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return annotations


def process_directory(directory_path, output_file, classes_file):
    with open(classes_file, 'r') as classes_file:
        classes = classes_file.read().splitlines()
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'class_label', 'class_name', 'x_center', 'y_center', 'width', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                annotations = read_bbox_annotations(file_path, classes)
                
                for annotation in annotations:
                    writer.writerow({
                        'filename': filename,
                        'class_label': annotation['class_label'],
                        'class_name': annotation['class_name'],
                        'x_center': annotation['x_center'],
                        'y_center': annotation['y_center'],
                        'width': annotation['width'],
                        'height': annotation['height']
                    })


# Exemplo de uso
directory_path = 'bboxes_yolo_exemple'
output_file = 'anotacoes.csv'
classes_file = 'classes.txt'
process_directory(directory_path, output_file, classes_file)
