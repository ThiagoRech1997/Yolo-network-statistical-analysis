import math
import scipy.stats as stats

def calcular_tamanho_amostra():
    # Solicitar os dados necessários
    desvio_padrao = float(input("Informe o desvio-padrão populacional (σ): "))
    margem_erro = float(input("Informe a margem de erro (E): "))
    grau_confianca = float(input("Informe o grau de confiança (em porcentagem): "))
    
    # Converter o grau de confiança para um valor crítico Zα/2
    valor_critico = stats.norm.ppf(1 - (1 - grau_confianca) / 2)
    print(valor_critico)
    # Calcular o tamanho mínimo da amostra
    n = math.ceil((valor_critico * desvio_padrao / margem_erro) ** 2)
    
    # Exibir o resultado
    print("O tamanho mínimo da amostra é:", n)

# Chamar a função para calcular o tamanho da amostra
calcular_tamanho_amostra()
