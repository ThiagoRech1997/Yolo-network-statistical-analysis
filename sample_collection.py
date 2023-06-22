import os
import glob
import random
from shutil import copyfile

def coletar_amostras(diretorio_origem, diretorio_destino):
    # Listar todos os arquivos PNG no diretório de origem
    arquivos_png = glob.glob(os.path.join(diretorio_origem, "*.png"))
    
    # Criar o diretório de destino se ele não existir
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)
    
    # Percorrer o período de 1 a 12 horas
    for hora in range(1, 13):
        # Filtrar as imagens para a hora correspondente
        imagens_hora = [arquivo for arquivo in arquivos_png if f"inferencias_exemple/Inferencia_{hora:02}*" in arquivo]
        # Selecionar aleatoriamente 22 imagens da hora correspondente
        amostra_hora = random.sample(imagens_hora, 22)
        
        # Salvar as imagens da amostra no diretório de destino
        for imagem in amostra_hora:
            nome_arquivo = os.path.basename(imagem)
            novo_nome = f"Amostra_Inferencia_Jun_{hora:02}h_{nome_arquivo}"
            caminho_destino = os.path.join(diretorio_destino, novo_nome)
            copyfile(imagem, caminho_destino)
        
        print(f"Amostras coletadas para a hora {hora:02}h")
    
    print("Amostras coletadas com sucesso!")

# Exemplo de uso
diretorio_origem = "inferencias_exemple"  # Defina o diretório de origem das imagens
diretorio_destino = "destino_exemple"  # Defina o diretório de destino das amostras
coletar_amostras(diretorio_origem, diretorio_destino)
