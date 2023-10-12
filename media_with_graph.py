import csv
from collections import Counter
import matplotlib.pyplot as plt

def calculate_class_frequency(csv_file):
    class_frequencies = Counter()
    total_records = 0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_label = int(row['class_label'])
            class_frequencies[class_label] += 1
            total_records += 1

    if total_records > 0:
        class_frequencies = {class_label: frequency / total_records for class_label, frequency in class_frequencies.items()}
        return class_frequencies
    else:
        return None

def visualize_class_frequency(class_frequencies):
    class_labels = list(class_frequencies.keys())
    frequencies = list(class_frequencies.values())

    print(frequencies)

    plt.bar(class_labels, frequencies)
    plt.xlabel('Classe')
    plt.ylabel('Frequência')
    plt.title('Frequência das Classes')
    plt.show()

# Exemplo de uso
csv_file = 'anotacoes.csv'
class_frequencies = calculate_class_frequency(csv_file)

if class_frequencies is not None:
    visualize_class_frequency(class_frequencies)
else:
    print("Nenhum dado encontrado no arquivo CSV.")
