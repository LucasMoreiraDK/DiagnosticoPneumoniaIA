# Importação das bibliotecas necessárias
from tensorflow.keras.models import load_model  # Para carregar o modelo de IA
import cv2  # OpenCV para processamento de imagens
import numpy as np  # Para cálculos numéricos
import os  # Para operações com sistema de arquivos
import tkinter as tk  # Para criar a interface gráfica
from tkinter import filedialog, messagebox  # Para caixas de diálogo e mensagens
from PIL import Image, ImageTk  # Para manipulação e exibição de imagens
from pathlib import Path  # Para manipulação segura de caminhos de arquivos

# Configura o NumPy para não usar notação científica ao exibir números
np.set_printoptions(suppress=True)

# 1. Carrega o modelo de IA pré-treinado
model_path = r"C:\iamodel\keras_model.h5"  # Caminho para o arquivo do modelo
if not os.path.exists(model_path):  # Verifica se o arquivo existe
    messagebox.showerror("Erro", f"Modelo não encontrado em {model_path}")  # Exibe mensagem de erro
    exit()  # Encerra o programa se o modelo não for encontrado

model = load_model(model_path, compile=False)  # Carrega o modelo sem compilar (para inferência)

# 2. Configuração das classes e cores para visualização
class_names = ['PNEUMONIA', 'NORMAL']  # Nomes das classes que o modelo pode prever
colors = {'PNEUMONIA': (0, 0, 255), 'NORMAL': (0, 255, 0)}  # Cores (BGR) para cada classe

# Classe principal da aplicação
class PneumoniaApp:
    def __init__(self, root):
        # Configuração inicial da janela principal
        self.root = root
        self.root.title("Diagnóstico de Pneumonia")  # Título da janela
        self.root.geometry("900x700")  # Tamanho da janela
        
        # Cria o frame principal
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=20)  # Adiciona padding
        
        # Adiciona o título da aplicação
        self.title_label = tk.Label(
            self.main_frame, 
            text="Sistema de Diagnóstico de Pneumonia", 
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=10)
        
        # Cria um frame para exibir a imagem
        self.image_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN)  # Borda afundada
        self.image_frame.pack(pady=10)
        
        # Label que vai conter a imagem
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        
        # Botão para carregar imagens
        self.load_button = tk.Button(
            self.main_frame, 
            text="Selecionar Imagem", 
            command=self.load_image,  # Define a função a ser chamada quando clicado
            font=("Helvetica", 12),
            bg="#4CAF50",  # Cor de fundo verde
            fg="white"  # Cor do texto branco
        )
        self.load_button.pack(pady=20)
        
        # Frame para exibir os resultados
        self.result_frame = tk.Frame(self.main_frame)
        self.result_frame.pack(pady=10)
        
        # Label para mostrar o diagnóstico
        self.diagnosis_label = tk.Label(
            self.result_frame, 
            text="Diagnóstico: ", 
            font=("Helvetica", 14)
        )
        self.diagnosis_label.pack()
        
        # Label para mostrar a confiança da predição
        self.confidence_label = tk.Label(
            self.result_frame, 
            text="Confiança: ", 
            font=("Helvetica", 14)
        )
        self.confidence_label.pack()
        
        # Variáveis para armazenar estado
        self.display_img = None  # Armazenará a imagem processada
        self.img_path = None  # Armazenará o caminho da imagem
    
    def load_image(self):
        # Abre a janela de seleção de arquivo
        initial_dir = os.path.expanduser('~')  # Começa no diretório do usuário
        self.img_path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png")],  # Filtra por tipos de imagem
            initialdir=initial_dir  # Diretório inicial
        )
        
        # Se o usuário cancelar a seleção
        if not self.img_path:
            return
            
        try:
            # Converte o caminho para objeto Path (mais seguro)
            img_path = Path(self.img_path)
            
            # Verifica se o arquivo existe
            if not img_path.exists():
                messagebox.showerror("Erro", f"Arquivo não encontrado: {img_path}")
                return
                
            # Carrega a imagem usando o caminho absoluto como string
            img = cv2.imread(str(img_path.absolute()))
            
            # Verifica se a imagem foi carregada corretamente
            if img is None:
                messagebox.showerror("Erro", "Falha ao decodificar a imagem. Formato não suportado ou arquivo corrompido.")
                return
                
            # Chama o processamento da imagem
            self.process_image(img)
            
        except Exception as e:
            # Mostra mensagem de erro genérico
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")
    
    def process_image(self, img):
        try:
            # Pré-processamento da imagem para o modelo
            
            # 1. Redimensiona para 224x224 pixels (tamanho esperado pelo modelo)
            input_img = cv2.resize(img, (224, 224))
            
            # 2. Converte para array NumPy e remodela para (1, 224, 224, 3)
            input_img = np.asarray(input_img, dtype=np.float32).reshape(1, 224, 224, 3)
            
            # 3. Normaliza os valores dos pixels para [-1, 1]
            input_img = (input_img / 127.5) - 1
            
            # Faz a predição usando o modelo
            prediction = model.predict(input_img)
            
            # Obtém o índice da classe com maior probabilidade
            index = np.argmax(prediction)
            
            # Obtém o nome da classe predita
            class_name = class_names[index]
            
            # Obtém o score de confiança (probabilidade)
            confidence_score = prediction[0][index]
            
            # Formata os textos para exibição
            diagnostico = f"Diagnostico: {class_name}"
            confianca = f"Confianca: {confidence_score*100:.2f}%"
            
            # Atualiza os labels na interface
            self.diagnosis_label.config(text=diagnostico)
            self.confidence_label.config(text=confianca)
            
            # Prepara a imagem para exibição
            
            # 1. Faz uma cópia da imagem original
            display_img = img.copy()
            
            # 2. Obtém a cor da borda baseada no diagnóstico
            border_color = colors[class_name]
            
            # 3. Adiciona uma borda colorida de 15px
            display_img = cv2.copyMakeBorder(
                display_img, 15, 15, 15, 15,  # Tamanho da borda (top, bottom, left, right)
                cv2.BORDER_CONSTANT,  # Tipo de borda (constante)
                value=border_color  # Cor da borda
            )
            
            # 4. Adiciona textos informativos na imagem
            
            # Define a fonte
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Adiciona o texto do diagnóstico
            cv2.putText(display_img, diagnostico, (30, 40), font, 0.9, (255, 255, 255), 2)
            
            # Adiciona o texto da confiança
            cv2.putText(display_img, confianca, (30, 80), font, 0.9, (255, 255, 255), 2)
            
            # Redimensiona a imagem mantendo a proporção
            
            # Obtém as dimensões originais
            height, width = display_img.shape[:2]
            
            # Define a altura máxima desejada
            max_height = 500
            
            # Calcula o fator de escala
            scale = max_height / height
            
            # Aplica o redimensionamento
            display_img = cv2.resize(display_img, (int(width*scale), int(height*scale)))
            
            # Converte a imagem para o formato que o tkinter pode exibir
            
            # 1. Converte de BGR (OpenCV) para RGB (PIL/Tkinter)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # 2. Converte para imagem PIL
            img_pil = Image.fromarray(display_img)
            
            # 3. Converte para formato PhotoImage do Tkinter
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Atualiza a imagem na interface
            
            # 1. Configura a imagem no label
            self.image_label.config(image=img_tk)
            
            # 2. Mantém uma referência para evitar garbage collection
            self.image_label.image = img_tk
            
        except Exception as e:
            # Mostra mensagem de erro se algo der errado no processamento
            messagebox.showerror("Erro", f"Erro no processamento: {str(e)}")

# Ponto de entrada da aplicação
if __name__ == "__main__":
    root = tk.Tk()  # Cria a janela principal
    app = PneumoniaApp(root)  # Instancia a aplicação
    root.mainloop()  # Inicia o loop principal da interface