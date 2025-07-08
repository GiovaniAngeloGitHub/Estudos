from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

FILE_PATH = Path("data/llb11_00249_2018_05_05_06_01_31.wav")
y, sr = librosa.load(FILE_PATH)

frequencies, times, Zxx = signal.stft(y, fs=sr, nperseg=1024, noverlap=512)

Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # +1e-10 para evitar log(0)

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, Zxx_db, shading='gouraud', cmap='viridis')
plt.colorbar(label='Intensidade (dB)')
plt.ylabel('Frequência [Hz]')
plt.xlabel('Tempo [s]')
plt.title('Espectrograma (SciPy)')
plt.ylim(0, 10000)  # Limitar eixo Y
plt.savefig(FILE_PATH.with_name("espectrograma.png"), dpi=300, bbox_inches='tight')
plt.show()