"""
Ses İşaretlerinde VAD (Voice Activity Detection) ve Voiced/Unvoiced Analizi
Tam ve Eksiksiz Sürüm - 2024
"""

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming  # Hata veren kısım düzeltildi
import matplotlib.pyplot as plt
import os

class AudioVADAnalyzer:
    """
    Ses sinyali üzerinde VAD ve Voiced/Unvoiced analizi yapan sınıf
    """
    
    def __init__(self, audio_file, frame_duration=0.02, overlap_ratio=0.5):
        """
        Başlangıç parametreleri
        """
        self.audio_file = audio_file
        self.frame_duration = frame_duration
        self.overlap_ratio = overlap_ratio
        
        # Ses dosyasını yükle
        self.sample_rate, self.signal = wavfile.read(audio_file)
        
        # Stereo ise mono'ya çevir
        if len(self.signal.shape) > 1:
            self.signal = np.mean(self.signal, axis=1)
        
        # Sinyali normalize et [-1, 1] aralığına
        self.normalized_signal = self._normalize_signal(self.signal)
        
        # Pencere parametrelerini hesapla
        self.frame_length = int(self.sample_rate * frame_duration)
        self.overlap_length = int(self.frame_length * overlap_ratio)
        self.hop_length = self.frame_length - self.overlap_length
        
        # Hamming penceresi oluştur
        self.window = hamming(self.frame_length)
        
        # Sonuçları saklamak için değişkenler
        self.frames = None
        self.energies = None
        self.zcr_values = None
        self.vad_mask = None
        self.voiced_mask = None
        
    def _normalize_signal(self, signal):
        """Sinyali [-1, 1] aralığına normalize et"""
        if signal.dtype == np.int16:
            max_val = 32768.0
        elif signal.dtype == np.int32:
            max_val = 2147483648.0
        else:
            max_val = np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 1.0
        
        normalized = signal.astype(np.float32) / max_val
        return np.clip(normalized, -1.0, 1.0)
    
    def _create_frames(self):
        """Sinyali pencerelerine ayır ve Hamming penceresi uygula"""
        signal = self.normalized_signal
        num_samples = len(signal)
        
        num_frames = int((num_samples - self.frame_length) / self.hop_length) + 1
        frames = np.zeros((num_frames, self.frame_length))
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end <= num_samples:
                frames[i] = signal[start:end] * self.window
        
        self.frames = frames
        return frames
    
    def _calculate_energy(self, frames):
        """Her pencere için karesel enerji hesapla"""
        energies = np.sum(frames ** 2, axis=1)
        self.energies = energies
        return energies
    
    def _calculate_zcr(self, frames):
        """Her pencere için Sıfır Geçiş Oranı (ZCR) hesapla"""
        zcr_values = []
        for frame in frames:
            sign_changes = np.diff(np.sign(frame))
            zero_crossings = np.sum(np.abs(sign_changes) > 0)
            zcr = zero_crossings / (2 * len(frame))
            zcr_values.append(zcr)
        
        self.zcr_values = np.array(zcr_values)
        return self.zcr_values
    
    def _detect_vad(self, energies, noise_duration=0.2, hangover_frames=3):
        """VAD - Konuşma bölgelerini tespit et"""
        noise_frames = int(noise_duration / (self.frame_duration * (1 - self.overlap_ratio)))
        noise_frames = max(1, min(noise_frames, len(energies)))
        
        noise_energy = energies[:noise_frames]
        noise_mean = np.mean(noise_energy)
        noise_std = np.std(noise_energy)
        
        threshold = noise_mean + 2 * noise_std
        vad_mask = energies > threshold
        
        # Hangover mekanizması
        for i in range(len(vad_mask)):
            if not vad_mask[i] and i > 0:
                prev_active = np.sum(vad_mask[max(0, i - hangover_frames):i])
                if prev_active >= hangover_frames - 1:
                    vad_mask[i] = True
        
        self.vad_mask = vad_mask
        return vad_mask
    
    def _classify_voiced_unvoiced(self, frames, energies, zcr_values, vad_mask):
        """Voiced/Unvoiced sınıflandırma"""
        classification = np.zeros(len(frames))
        if not np.any(vad_mask):
            return classification
            
        speech_energies = energies[vad_mask]
        speech_zcr = zcr_values[vad_mask]
        
        energy_threshold = np.percentile(speech_energies, 30)
        zcr_threshold = np.percentile(speech_zcr, 50)
        
        for i in range(len(frames)):
            if vad_mask[i]:
                if zcr_values[i] < zcr_threshold and energies[i] > energy_threshold:
                    classification[i] = 2  # Voiced
                else:
                    classification[i] = 1  # Unvoiced
        
        self.voiced_mask = classification
        return classification
    
    def save_vad_audio(self, output_file="vad_output.wav"):
        """VAD uygulanmış sesi kaydet"""
        signal_vad_mask = np.zeros(len(self.normalized_signal), dtype=bool)
        for i, is_speech in enumerate(self.vad_mask):
            if is_speech:
                start = i * self.hop_length
                end = min(start + self.frame_length, len(self.normalized_signal))
                signal_vad_mask[start:end] = True
        
        speech_signal = self.signal[signal_vad_mask]
        wavfile.write(output_file, self.sample_rate, speech_signal.astype(self.signal.dtype))
        print(f"• Temizlenmiş ses kaydedildi: {output_file}")

    def _print_statistics(self):
        """İstatistikleri yazdır"""
        total = len(self.frames)
        speech = np.sum(self.vad_mask)
        voiced = np.sum(self.voiced_mask == 2)
        unvoiced = np.sum(self.voiced_mask == 1)
        print(f"\n📈 Analiz İstatistikleri:")
        print(f"• Konuşma Pencereleri: {speech}/{total}")
        print(f"• Voiced (Sesli): {voiced} | Unvoiced (Sessiz): {unvoiced}")

    def visualize_results(self):
        """Sonuçları görselleştir"""
        time = np.arange(len(self.normalized_signal)) / self.sample_rate
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Sinyal
        axes[0].plot(time, self.normalized_signal, color='steelblue')
        axes[0].set_title('Normalize Sinyal')
        
        # Enerji/ZCR
        frame_time = np.arange(len(self.frames)) * self.hop_length / self.sample_rate
        axes[1].plot(frame_time, self.energies, 'b', label='Enerji')
        ax2_twin = axes[1].twinx()
        ax2_twin.plot(frame_time, self.zcr_values, 'r', label='ZCR')
        axes[1].set_title('Enerji ve ZCR')
        
        # VAD Maskeleri
        axes[2].plot(time, self.normalized_signal, color='gray', alpha=0.3)
        for i, val in enumerate(self.voiced_mask):
            t_start = i * self.hop_length / self.sample_rate
            t_end = t_start + (self.frame_length / self.sample_rate)
            if val == 2: axes[2].axvspan(t_start, t_end, color='green', alpha=0.4)
            elif val == 1: axes[2].axvspan(t_start, t_end, color='yellow', alpha=0.4)
        axes[2].set_title('Yeşil: Sesli (Voiced) | Sarı: Sessiz Harf (Unvoiced)')
        
        plt.tight_layout()
        plt.show()
        self._print_statistics()

    def analyze(self):
        """Tüm süreci çalıştır"""
        print(f"🎤 İşlem başlıyor: {self.audio_file}")
        self._create_frames()
        self._calculate_energy(self.frames)
        self._calculate_zcr(self.frames)
        self._detect_vad(self.energies)
        self._classify_voiced_unvoiced(self.frames, self.energies, self.zcr_values, self.vad_mask)
        print("✅ Analiz tamamlandı.")

# --- ANA ÇALIŞTIRMA ---
if __name__ == "__main__":
    audio_path = "test_audio.wav" 
    
    if os.path.exists(audio_path):
        analyzer = AudioVADAnalyzer(audio_path)
        analyzer.analyze()
        analyzer.save_vad_audio("sonuc.wav")
        analyzer.visualize_results()
    else:
        print(f"Hata: {audio_path} dosyası bulunamadı!")