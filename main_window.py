"""
    @authors:
        Luiz Felipe Gonçalves Barbosa Viana - 11611ECP021
        Matheus Cleber Silva Guerra - 11721ECP009
        Pedro Henrique Rodrigues Marques Dos Santos - 11611ECP017
    @def: Trabalho final de Sinais e Multimidia.
"""

import sys, cv2, numpy as np, imutils, matplotlib.pyplot as plt, scipy.ndimage

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


# Kernels utilizado com base no programa criado pelo professor
kernel={
    0: np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
    1: np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=float),
    2: np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    3: np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    4: np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=float),
    5: np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]], dtype=float),
    6: np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
    7: np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    8: np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    9: np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    10: np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float),
    11: np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),
    12: (1/9)*np.ones((3,3), dtype=float),
    13: (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
    14: (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
    15: np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    16: (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

"""
    @def: Classe principal
"""
class MainWindow(QWidget):

    def __init__(self):

        # Inicializo a aplicação baseado no meu layout "home.ui"
        super().__init__()
        self.dlg = uic.loadUi("home.ui")
        self.dlg.show()

        # Inicializo um timer para controlar meus frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.initialize_camera)

        # Inicializo o evento de clicar para abrir a camêra
        self.dlg.qbutton_start.clicked.connect(self.timer_control)


    # @def: Inicialização da Câmera
    def initialize_camera(self):

        # Recupero o valor de tamanho do meu width
        width_camera = self.dlg.qlabel_camera.width()
        
        # Recupero o frame lido
        ret, image = self.cap.read()
        
        # Realizo um resize para o tamanho da imagem no aplicacao
        image = imutils.resize(image, width=width_camera)

        # Transformo a imagem em cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Faço a convolucao
        conv = np.uint8(np.round(convolution(gray, kernel[self.dlg.qcombobox_conv.currentIndex()])))

        # Recupero os tamanhos -> defino o meu canal -> calculo o meu passo
        height, width = conv.shape
        channel = 1
        step = channel * width
        
        qImg = QImage(conv.data.tobytes(), width, height, step, QImage.Format_Indexed8)
        
        # Se meu ativar blur estiver clicado, eu passo a imagem no meu detector de blur
        if self.dlg.qradiobutton_blur.isChecked():
            (mean, blurry) = blur_detect(gray, size=60, thresh=int(self.dlg.qlineedit_th.text()), showing=False)
            text = "Resultado: " + "Está borrado" if blurry else "Não está borrado"
            
            # Escrevo meu resultado
            self.dlg.qlabel_resultado.setText(str(text) + " - " + str(mean))
            
        else:
            # Escrevo meu resultado
            self.dlg.qlabel_resultado.setText('Resultado: Blur não selecionado')
        
        # Exibo a imagem apos o tratamento
        self.dlg.qlabel_camera.setPixmap(QPixmap.fromImage(qImg))
        self.dlg.qlabel_camera.setScaledContents(True)

    def timer_control(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)


# @def: Convolução

def my_convolve2d(a, conv_filter):
    submatrices = np.array([
         [a[:-2,:-2], a[:-2,1:-1], a[:-2,2:]],
         [a[1:-1,:-2], a[1:-1,1:-1], a[1:-1,2:]],
         [a[2:,:-2], a[2:,1:-1], a[2:,2:]]])
    multiplied_subs = np.einsum('ij,ijkl->ijkl',conv_filter,submatrices)
    return np.sum(np.sum(multiplied_subs, axis = -3), axis = -3)

def convolution(im, omega):

    return scipy.ndimage.convolve(im, omega, output=None, mode='reflect', cval=0.0, origin=0)


# @def: Deteção de Blur
def blur_detect(image, size=60, thresh=10, showing=False):

    # recupero as dimensões da imagem e derivo as coordenadas do centro (x, y) dividindo por dois
    (h, w) = image.shape
    (deltaX, deltaY) = (int(w / 2.0), int(h / 2.0))
    
    # Calculo a FFT -> encontro a transformação de frequência -> mudo componente de frequência zero para o centro
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # verifique se estamos vendo nossa saida
    if showing:

        # calcular a magnitude da transformada
        magnitude = 20 * np.log(np.abs(fftShift))

        # Entrada original
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Imagem de magnitude
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # Exibir com o show da matplotlib
        plt.show()

    # remover frequências baixas -> aplicar o deslocamento inverso -> aplicar o FFT inverso
    fftShift[deltaY - size:deltaY + size, deltaX - size:deltaX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # Calculo o espectro de magnitude da imagem reconstruída -> Calcule a média dos valores de magnitude
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    #se o valor médio das magnitudes for menor que o valor limite <-> imagem borrada
    return (mean, mean <= thresh)


# Inicializando a aplicação
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Inicializo a MainWindow
    mainWindow = MainWindow()

    # Finalizo a aplicacao
    sys.exit(app.exec_())