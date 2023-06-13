import csv

def calculate_average(csv_file):
    total_records = 0
    sum_x_center = 0.0
    sum_y_center = 0.0
    sum_width = 0.0
    sum_height = 0.0
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            total_records += 1
            sum_x_center += float(row['x_center'])
            sum_y_center += float(row['y_center'])
            sum_width += float(row['width'])
            sum_height += float(row['height'])
    
    if total_records > 0:
        average_x_center = sum_x_center / total_records
        average_y_center = sum_y_center / total_records
        average_width = sum_width / total_records
        average_height = sum_height / total_records
        
        return {
            'average_x_center': average_x_center,
            'average_y_center': average_y_center,
            'average_width': average_width,
            'average_height': average_height
        }
    else:
        return None


# Exemplo de uso
csv_file = 'anotacoes.csv'
average_data = calculate_average(csv_file)

if average_data is not None:
    print(f"Average x_center: {average_data['average_x_center']}")
    print(f"Average y_center: {average_data['average_y_center']}")
    print(f"Average width: {average_data['average_width']}")
    print(f"Average height: {average_data['average_height']}")
else:
    print("Nenhum dado encontrado no arquivo CSV.")
