def read_bbox_annotations(file_path):
    annotations = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip().split()
            
            # Extrair informações das anotações
            class_label = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])
            
            annotations.append({
                'class_label': class_label,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return annotations

# Exemplo de uso
annotations = read_bbox_annotations('annotations.txt')
for annotation in annotations:
    print(annotation)
