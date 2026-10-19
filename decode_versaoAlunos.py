# Projeto 6 - DTMF - lado receptor
from suaBibSignal import *
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time

try:
    import peakutils
except ImportError:
    peakutils = None
    from scipy.signal import find_peaks   # alternativa se peakutils não estiver instalado

# Mesma tabela do emissor
LINHAS = [697, 770, 852, 941]
COLUNAS = [1209, 1336, 1477, 1633]
TECLAS = [['1', '2', '3', 'A'],
          ['4', '5', '6', 'B'],
          ['7', '8', '9', 'C'],
          ['X', '0', '#', 'D']]   # X = '*'
TOL = 10   # tolerância em Hz para casar pico com a tabela

def todB(s):
    return 10 * np.log10(s)


def achar_picos(xf, yf, minimo=5):
    """Ajusta a sensibilidade (thres) até ter ao menos 'minimo' picos.
    Picos a menos de 5 Hz de um pico mais forte são descartados (min_dist)."""
    passo_hz = xf[1] - xf[0]
    min_dist = max(1, int(5 / passo_hz))
    thres = 0.5
    while thres > 0.0001:
        if peakutils is not None:
            idx = peakutils.indexes(yf, thres=thres, min_dist=min_dist)
        else:
            idx, _ = find_peaks(yf, height=thres * np.max(yf), distance=min_dist)
        if len(idx) >= minimo:
            break
        thres *= 0.7
    idx = sorted(idx, key=lambda i: yf[i], reverse=True)   # do maior para o menor
    return np.array(idx)


def identificar_tecla(xf, yf, idx):
    """Entre os picos, pega o mais forte que casa com uma freq. de linha e
    o mais forte que casa com uma freq. de coluna."""
    freqs = [xf[i] for i in idx]          # já ordenadas por amplitude
    linha = next((l for f in freqs for l in LINHAS if abs(f - l) <= TOL), None)
    coluna = next((c for f in freqs for c in COLUNAS if abs(f - c) <= TOL), None)
    if linha is None or coluna is None:
        return None, linha, coluna
    return TECLAS[LINHAS.index(linha)][COLUNAS.index(coluna)], linha, coluna


def main():
    signal = signalMeu()

    fs = 44100                 # taxa de amostragem
    sd.default.samplerate = fs
    sd.default.channels = 1    # se der problema, o driver pode exigir 2 (trate abaixo)
    duration = 3               # segundos de gravação
    numAmostras = int(duration * fs)

    print("A captação começará em 3 segundos. Comece a emitir o tom agora!")
    time.sleep(3)
    print("Gravação inicializada...")
    audio = sd.rec(numAmostras, fs, channels=1)
    sd.wait()
    print("...     FIM")

    # audio pode vir como matriz (N, canais): pega só a primeira coluna
    dados = audio[:, 0] if audio.ndim > 1 else audio
    t = np.linspace(0, duration, numAmostras, endpoint=False)

    # Sinal no tempo (gravação inteira)
    plt.figure()
    plt.plot(t, dados)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.title("Sinal recebido (gravação completa)")
    plt.grid(True)

    # Fourier
    xf, yf = signal.calcFFT(dados, fs)
    signal.plotFFT(dados, fs, "Fourier do sinal recebido")

    # Picos
    idx = achar_picos(xf, yf, minimo=5)
    print("Picos encontrados (Hz):")
    for i in idx[:6]:
        print("freq de pico sao {:.2f}  (amplitude {:.4f})".format(xf[i], yf[i]))

    tecla, linha, coluna = identificar_tecla(xf, yf, idx)
    if tecla is None:
        print("Não foi possível identificar a tecla. Aumente o volume / reduza ruído e tente de novo.")
    else:
        print("Frequências identificadas: {} Hz + {} Hz".format(linha, coluna))
        print("Tecla pressionada: {}".format('*' if tecla == 'X' else tecla))

    plt.show()


if __name__ == "__main__":
    main()