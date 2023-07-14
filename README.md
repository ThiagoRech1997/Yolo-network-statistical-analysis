# Algoritmos para avaliacao da eficassia de uma rede Yolo para deteccao de leveduras

-------------------------------
### Bibliotecas python para instalar:
```shell
sudo apt install python3-pip # Caso nao possua o pip instalado na maquina
pip install matplotlib
pip install scipy
```
-------------------------------
### Para dedinir o tamanho da amostra:
```shell
python3 ./n-population.py
```
para um exemplo utilizaremos o tamanho da populacao em 3600 capturas de imagens


### Seleciona Amostras de Imagens
```shell
python3 ./select_images_amostration.py
```

Altere as Linhas:
```python
source_directory = 'diretorio-com-as-imagens'
destination_directory = 'local-de-destino-para-imagens-selecionadas'
num_images_to_select = 22 # number of images to select
```

--------------------------------
### Faca as anotacoes da matri de confusao em qualquer softweare de anotacao de imagens

O Arquivo classes.txt contem as classes que nos interessam para as anotacoes, sendo elas:

VP = Verdadeiro Positivo

VN = Verdadeiro Negativo

FP = Falso Positivo

FN = Falso Negativo

--------------------------------

### Faca a leitura dos arquivos de anotacoes das imagens anotadas com:

```shell
python3 ./associate_class.py
```

Altere as Linhas:

```python
directory_path = 'diretorio-com-as-imagens-selecionadas-e-anotadas'
output_file = 'anotacoes.csv'
classes_file = 'classes.txt'
```

### Faca a estatistica das anotacoes com:

```shell
python3 ./media_with_graph.py
```

Caso no arquivo anterior tenha sido alterado o aquivo .csv, altere a linha:
```python
csv_file = 'anotacoes.csv'
```