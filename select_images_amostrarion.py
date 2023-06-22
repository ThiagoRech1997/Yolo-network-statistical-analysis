import os
import random
import shutil

def select_random_images(source_dir, destination_dir, num_images):
    # Lista todos os arquivos de imagem no diretório de origem
    image_files = [file for file in os.listdir(source_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Verifica se o número de imagens solicitado é maior que o número de imagens disponíveis
    num_images = min(num_images, len(image_files))
    
    # Seleciona aleatoriamente as imagens
    selected_images = random.sample(image_files, num_images)
    
    # Copia as imagens selecionadas para o diretório de destino
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(destination_dir, image)
        shutil.copyfile(source_path, destination_path)

# Exemplo de uso
source_directory = 'destino_exemple/inferencias'
destination_directory = 'destino_exemple/selecao'
num_images_to_select = 22

select_random_images(source_directory, destination_directory, num_images_to_select)
