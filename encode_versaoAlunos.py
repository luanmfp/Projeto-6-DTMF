# Projeto 6 - DTMF - lado emissor
from suaBibSignal import *
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sys

# Tabela DTMF: tecla -> (freq. baixa [linha], freq. alta [coluna])
# (o enunciado traz 679 e 825 Hz, que são erros de digitação: o padrão DTMF é 697 e 852 Hz,
#  e o exemplo de saída do próprio enunciado usa 1209 + 852 para o 7)
DTMF = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    'X': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633),
}

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def todB(s):
    return 10 * np.log10(s)


def main():
    print("Inicializando encoder")
    fs = 44100          # taxa de amostragem
    duracao = 3         # segundos emitindo (suficiente para o receptor gravar)
    A = 1               # amplitude de cada senoide
    sinal = signalMeu()

    print("Aguardando usuário")
    NUM = input("Digite uma tecla (0-9; também aceita A-D, # e X para *): ").strip().upper()
    if NUM == '*':
        NUM = 'X'
    if NUM not in DTMF:
        print("Tecla inválida.")
        sys.exit(1)

    print("Gerando Tons base")
    f1, f2 = DTMF[NUM]
    print("Gerando Tom referente ao símbolo : {}".format(NUM))
    print("Frequências : {} + {}".format(f1, f2))

    t, s1 = sinal.generateSin(f1, A, duracao, fs)
    _, s2 = sinal.generateSin(f2, A, duracao, fs)
    tone = s1 + s2      # soma das duas senoides (pico máximo = 2)
    tone = tone / np.max(np.abs(tone)) * 0.9   # normaliza para não saturar a placa de som

    print("Executando as senoides (emitindo o som)")
    sd.play(tone, fs)
    sd.wait()           # aguarda fim do áudio

    # (opcional) salva em arquivo
    try:
        from scipy.io import wavfile
        wavfile.write("dtmf_{}.wav".format(NUM if NUM != '#' else 'hash'), fs,
                      (tone * 32767).astype(np.int16))
    except Exception as e:
        print("Não foi possível salvar o wav:", e)

    print("Plotando")
    # Tempo: apenas alguns períodos (primeiras 500 amostras ~ 11 ms)
    plt.figure()
    plt.plot(t[:500] * 1000, tone[:500])
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Amplitude")
    plt.title("Sinal emitido - tecla {} ({} Hz + {} Hz)".format(NUM, f1, f2))
    plt.grid(True)

    # Frequência
    sinal.plotFFT(tone, fs, "Fourier do sinal emitido - tecla {}".format(NUM))
    plt.show()


if __name__ == "__main__":
    main()
