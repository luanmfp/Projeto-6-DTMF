import numpy as np
import matplotlib.pyplot as plt


class signalMeu:
    """Biblioteca de apoio: geração de senoides e Transformada de Fourier."""

    def generateSin(self, freq, amplitude, time, fs):
        """Retorna (t, s): senoide A*sin(2*pi*f*t) com 'time' segundos a 'fs' amostras/s."""
        n = int(round(time * fs))
        t = np.linspace(0, time, n, endpoint=False)
        s = amplitude * np.sin(2 * np.pi * freq * t)
        return t, s

    def calcFFT(self, signal, fs):
        """Retorna (xf, yf): frequências (Hz) e amplitude do espectro (só freq. positivas)."""
        signal = np.asarray(signal, dtype=float)
        N = len(signal)
        janela = np.hanning(N)                      # reduz vazamento espectral
        yf = np.abs(np.fft.rfft(signal * janela)) * 2 / np.sum(janela)
        xf = np.fft.rfftfreq(N, 1 / fs)
        return xf, yf

    def plotFFT(self, signal, fs, titulo="Fourier do sinal", fmax=2500):
        xf, yf = self.calcFFT(signal, fs)
        plt.figure()
        plt.plot(xf, yf)
        plt.xlim(0, fmax)
        plt.grid(True)
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Amplitude")
        plt.title(titulo)
        return xf, yf
