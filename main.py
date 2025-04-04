# Importação das bibliotecas necessárias
from tensorflow.keras.models import load_model  # Para carregar o modelo pré-treinado
import cv2  # OpenCV para processamento de imagens
import numpy as np  # Para cálculos numéricos
import os  # Para operações com sistema de arquivos

# Configura o NumPy para não usar notação científica ao exibir números
np.set_printoptions(suppress=True)

# 1. Verificação e carregamento do modelo de IA
model_path = r"C:\iamodel\keras_model.h5"  # Caminho absoluto para o arquivo do modelo
if not os.path.exists(model_path):  # Verifica se o arquivo do modelo existe
    print(f"Erro: Modelo não encontrado em {model_path}")
    exit()  # Encerra o programa se o modelo não for encontrado

model = load_model(model_path, compile=False)  # Carrega o modelo sem compilar (para inferência)

# 2. Definição das classes e esquema de cores para visualização
class_names = ['PNEUMONIA', 'NORMAL']  # Nomes das classes que o modelo pode prever
colors = {'PNEUMONIA': (0, 0, 255), 'NORMAL': (0, 255, 0)}  # Cores (BGR): Vermelho para pneumonia, Verde para normal

# 3. Carregamento e verificação da imagem de entrada
img_path = r'aduglade.jpg'  # Caminho para a imagem a ser analisada
if not os.path.exists(img_path):  # Verifica se o arquivo de imagem existe
    print(f"Erro: Imagem não encontrada em {img_path}")
    exit()  # Encerra se a imagem não for encontrada

img = cv2.imread(img_path)  # Carrega a imagem em formato BGR
if img is None:  # Verifica se a imagem foi carregada corretamente
    print("Erro: Falha ao decodificar a imagem")
    exit()  # Encerra se houver erro no carregamento

# 4. Pré-processamento da imagem para o modelo
try:
    # Redimensiona a imagem para 224x224 pixels (tamanho esperado pelo modelo)
    input_img = cv2.resize(img, (224, 224))
    
    # Converte para array NumPy e remodela para (1, 224, 224, 3) - (batch, height, width, channels)
    input_img = np.asarray(input_img, dtype=np.float32).reshape(1, 224, 224, 3)
    
    # Normaliza os pixels para o intervalo [-1, 1] (como o modelo foi treinado)
    input_img = (input_img / 127.5) - 1
except Exception as e:  # Captura qualquer erro durante o pré-processamento
    print(f"Erro no pré-processamento: {str(e)}")
    exit()  # Encerra o programa em caso de erro

# 5. Realização da predição com o modelo
try:
    # Executa a predição (forward pass) na imagem pré-processada
    prediction = model.predict(input_img)
    
    # Obtém o índice da classe com maior probabilidade
    index = np.argmax(prediction)
    
    # Obtém o nome da classe predita
    class_name = class_names[index]
    
    # Obtém o score de confiança (probabilidade) da predição
    confidence_score = prediction[0][index]
except Exception as e:  # Captura qualquer erro durante a predição
    print(f"Erro na predição: {str(e)}")
    exit()  # Encerra o programa em caso de erro

# 6. Formatação e exibição dos resultados no console
diagnostico = f"Diagnóstico: {class_name}"  # String formatada com o resultado
confianca = f"Confiança: {confidence_score*100:.2f}%"  # String com a confiança formatada

# Imprime os resultados com separadores visuais
print("\n" + "="*50)  # Linha decorativa
print(diagnostico)  # Exibe o diagnóstico
print(confianca)  # Exibe a confiança
print("="*50 + "\n")  # Linha decorativa

# 7. Preparação da imagem para visualização
display_img = img.copy()  # Cria uma cópia da imagem original para exibição

# Adiciona uma borda colorida de 15px de acordo com o diagnóstico
border_color = colors[class_name]  # Seleciona a cor baseada no resultado
display_img = cv2.copyMakeBorder(
    display_img, 15, 15, 15, 15,  # Bordas superior, inferior, esquerda, direita
    cv2.BORDER_CONSTANT,  # Tipo de borda (constante)
    value=border_color  # Cor da borda
)

# Adiciona textos informativos diretamente na imagem usando OpenCV
font = cv2.FONT_HERSHEY_SIMPLEX  # Define a fonte do texto
# Adiciona o texto do diagnóstico na posição (30,40)
cv2.putText(display_img, diagnostico, (30, 40), font, 0.9, (255, 255, 255), 2)
# Adiciona o texto da confiança na posição (30,80)
cv2.putText(display_img, confianca, (30, 80), font, 0.9, (255, 255, 255), 2)

# Redimensionamento inteligente mantendo a proporção original
height, width = display_img.shape[:2]  # Obtém as dimensões originais
max_size = 800  # Tamanho máximo (em pixels) para o maior lado
scale = max_size / max(height, width)  # Calcula o fator de escala
# Aplica o redimensionamento proporcional
display_img = cv2.resize(display_img, (int(width*scale), int(height*scale)))

# 8. Exibição final da imagem com resultados
cv2.imshow('Resultado - Diagnóstico de Pneumonia', display_img)  # Mostra a imagem
cv2.waitKey(0)  # Aguarda até que qualquer tecla seja pressionada
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV