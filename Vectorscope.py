import sys
import os
import numpy as np
import pygame
import time
import tempfile

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QSizePolicy, QVBoxLayout, QHBoxLayout,
    QSlider, QComboBox, QFileDialog, QPushButton, QCheckBox,
    QMainWindow, QGroupBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QRadialGradient, QBrush

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print(
        "pydub not available - limited audio format support. "
        "Install it with 'pip install pydub' for additional codecs."
    )

# === AUDIO PROCESSING THREAD ===
class AudioLoaderThread(QThread):
    loaded = pyqtSignal(object, str)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.loaded.emit(self.load_audio(), self.file_path)
        except Exception as e:
            self.error.emit(str(e))
    
    def load_audio(self):
        print(f"Loading audio file: {self.file_path}")
        
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(self.file_path)
                if audio.channels == 1:
                    audio = audio.set_channels(2)
                audio = audio.set_frame_rate(44100).set_sample_width(2)
                
                raw_data = audio.raw_data
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                audio_data = audio_array.reshape(-1, 2) / 32768.0
                
                # Create temp WAV
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                wav_path = temp_wav.name
                temp_wav.close()
                audio.export(wav_path, format="wav")
                
                return {'data': audio_data, 'wav_path': wav_path, 'temp': True}
            except Exception as e:
                print(f"pydub failed, trying fallback: {e}")
        
        # Fallback for WAV files
        if self.file_path.lower().endswith('.wav'):
            import wave
            with wave.open(self.file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()

                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_array = audio_array.reshape(-1, channels)

                if channels == 1:
                    audio_array = np.repeat(audio_array, 2, axis=1)

                if sample_rate != 44100:
                    resample_factor = 44100 / sample_rate
                    new_length = int(audio_array.shape[0] * resample_factor)
                    indices = np.linspace(0, audio_array.shape[0] - 1, new_length)
                    resampled = []
                    for ch in range(audio_array.shape[1]):
                        resampled.append(
                            np.interp(indices, np.arange(audio_array.shape[0]), audio_array[:, ch])
                        )
                    audio_array = np.stack(resampled, axis=1).astype(np.int16)

                audio_data = audio_array.reshape(-1, 2) / 32768.0
                return {'data': audio_data, 'wav_path': self.file_path, 'temp': False}
        
        if PYDUB_AVAILABLE:
            raise Exception("Unsupported audio format")
        raise Exception(
            "Unsupported audio format. Install pydub for broader file support."
        )

# === SHADER-LIKE EFFECTS ===
class ShaderEffects:
    @staticmethod
    def apply_bloom(painter, points, hue, intensity, bloom_radius=20):
        """Apply bloom/glow effect to points"""
        if not points:
            return
        
        # Draw multiple layers of glow
        for layer in range(3):
            radius = bloom_radius * (3 - layer)
            alpha = int(intensity * (0.3 / (layer + 1)))
            
            for point in points[-10:]:  # Only recent points
                gradient = QRadialGradient(point, radius)
                color = QColor.fromHsv(hue, 200, 255, alpha)
                gradient.setColorAt(0, color)
                gradient.setColorAt(1, QColor(0, 0, 0, 0))
                
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(point, radius, radius)
    
    @staticmethod
    def draw_phosphor_trail(painter, path, hue, trail_alpha, thickness=2.0):
        """Draw phosphor-like trailing effect"""
        if len(path) < 2:
            return
        
        trail_len = len(path)
        for i in range(1, trail_len):
            age = i / trail_len
            
            # Phosphor decay simulation
            fade_alpha = int((1.0 - age) * trail_alpha * np.exp(-age * 2))
            if fade_alpha < 5:
                continue
            
            # Width varies with age
            width = thickness * (1.0 - age * 0.8)
            
            # Color shifts slightly as it fades
            hue_shift = int(age * 20)
            color = QColor.fromHsv((hue + hue_shift) % 360, 255 - int(age * 100), 255, fade_alpha)
            
            pen = QPen(color)
            pen.setWidthF(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(path[i - 1], path[i])
    
    @staticmethod
    def draw_beam_core(painter, path, intensity):
        """Draw bright beam core"""
        if len(path) < 2:
            return
        
        # Draw thin bright core
        core_pen = QPen(QColor(255, 255, 255, min(255, intensity + 100)))
        core_pen.setWidthF(1.0)
        painter.setPen(core_pen)
        
        for i in range(max(0, len(path) - 20), len(path)):
            if i > 0:
                painter.drawLine(path[i - 1], path[i])

# === AUDIO ENGINE ===
def generate_wave(wave_type, freq, amp, samplerate=44100, duration=1.0):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    if wave_type == 'sine':
        return amp * np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == 'triangle':
        return amp * 2 * np.arcsin(np.sin(2 * np.pi * freq * t)) / np.pi
    elif wave_type == 'sawtooth':
        return amp * 2 * (t * freq - np.floor(0.5 + t * freq))
    else:
        return np.zeros_like(t)

class AudioOutput:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.left_freq = 440
        self.right_freq = 440
        self.left_wave = 'sine'
        self.right_wave = 'sine'
        self.left_amp = 0.2
        self.right_amp = 0.2
        self.last_params = None
        self.channel = None
        self.latest_stereo = None
        self.is_file_mode = False
        self.is_muted = False
        
        self.file_audio_data = None
        self.file_start_time = None
        self.file_sample_rate = 44100
        self.current_file_path = None
        self.temp_files = []
        
        self.generate_continuous_buffer()

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_update_needed)
        self.timer.start(33)

    def check_update_needed(self):
        if self.is_file_mode and self.file_audio_data is not None:
            self.update_file_playback()
        else:
            current_params = (self.left_wave, self.left_freq, self.left_amp, 
                            self.right_wave, self.right_freq, self.right_amp)
            if current_params != self.last_params:
                self.generate_continuous_buffer()
    
    def update_file_playback(self):
        if self.file_audio_data is None:
            return
        
        if self.file_start_time is None:
            self.file_start_time = time.time()
        
        elapsed = time.time() - self.file_start_time
        total_duration = len(self.file_audio_data) / self.file_sample_rate
        playback_position = elapsed % total_duration
        sample_position = int(playback_position * self.file_sample_rate)
        
        window_size = 2048
        start_pos = sample_position
        end_pos = start_pos + window_size
        
        if end_pos > len(self.file_audio_data):
            part1 = self.file_audio_data[start_pos:]
            needed = window_size - len(part1)
            part2 = self.file_audio_data[:needed] if needed > 0 else np.empty((0, 2))
            self.latest_stereo = np.vstack([part1, part2]) if len(part2) > 0 else part1
        else:
            self.latest_stereo = self.file_audio_data[start_pos:end_pos]

    def generate_continuous_buffer(self):
        if self.is_file_mode:
            return
        self.last_params = (self.left_wave, self.left_freq, self.left_amp, 
                          self.right_wave, self.right_freq, self.right_amp)
        duration = 10.0
        l = generate_wave(self.left_wave, self.left_freq, self.left_amp, duration=duration)
        r = generate_wave(self.right_wave, self.right_freq, self.right_amp, duration=duration)
        stereo = np.vstack((l, r)).T
        self.latest_stereo = stereo.copy()
        stereo_int = (stereo * 32767).astype(np.int16)
        new_sound = pygame.sndarray.make_sound(stereo_int.copy())
        
        if self.channel:
            self.channel.stop()
        self.channel = new_sound.play(loops=-1)
        if self.is_muted:
            self.channel.set_volume(0, 0)

    def set_file_audio(self, audio_info, file_path):
        """Load audio file data"""
        self.is_file_mode = True
        if self.channel:
            self.channel.stop()
        
        self.file_audio_data = audio_info['data']
        self.file_sample_rate = 44100
        self.file_start_time = time.time()
        self.current_file_path = file_path
        
        if audio_info['temp']:
            self.temp_files.append(audio_info['wav_path'])
        
        # Play the audio
        sound = pygame.mixer.Sound(audio_info['wav_path'])
        self.channel = sound.play(loops=-1)
        if self.is_muted:
            self.channel.set_volume(0, 0)
        
        self.latest_stereo = self.file_audio_data[:2048].copy()
    
    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.channel:
            if self.is_muted:
                self.channel.set_volume(0, 0)
            else:
                self.channel.set_volume(self.left_amp, self.right_amp)
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

# === MAIN WINDOW ===
class VectorscopeWidget(QLabel):
    """Custom widget for vectorscope display with shader effects"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black; border: 2px solid #333;")
        
        # Display parameters
        self.trail_alpha = 180
        self.glow_intensity = 150
        self.bloom_radius = 15
        self.x_scale = 0.45
        self.y_scale = 0.45
        self.hue = 120
        self.invert_y = False
        self.phosphor_decay = True
        self.beam_width = 2.0
        
        # Path history
        self.path_history = []
        self.max_path_length = 500

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio = AudioOutput()
        self.loader_thread = None
        
        self.setWindowTitle("Enhanced Vectorscope")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a2a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #666;
                border: 1px solid #555;
                width: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
            QComboBox {
                background-color: #2a2a2a;
                border: 1px solid #555;
                padding: 3px;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Top controls
        self.create_top_controls(main_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Vectorscope
        self.scope = VectorscopeWidget()
        content_layout.addWidget(self.scope, 3)
        
        # Side controls
        self.create_side_controls(content_layout)
        
        main_layout.addLayout(content_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scope)
        self.timer.start(16)  # 60 FPS for smooth animation

    def closeEvent(self, event):
        """Handle window close and clean up resources."""
        self.audio.cleanup()
        pygame.mixer.quit()
        super().closeEvent(event)

    def create_top_controls(self, parent_layout):
        """Create top control panel"""
        top_group = QGroupBox("Audio Source")
        top_layout = QHBoxLayout()
        
        # Mode selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Tone Generator", "Audio File"])
        self.mode_selector.currentTextChanged.connect(self.switch_audio_mode)
        
        # File controls
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio_file)
        self.load_button.setEnabled(False)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888;")
        
        # Mute button
        self.mute_button = QPushButton("Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.clicked.connect(self.audio.toggle_mute)
        
        top_layout.addWidget(QLabel("Mode:"))
        top_layout.addWidget(self.mode_selector)
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.file_label)
        top_layout.addStretch()
        top_layout.addWidget(self.mute_button)
        
        top_group.setLayout(top_layout)
        parent_layout.addWidget(top_group)

    def create_side_controls(self, parent_layout):
        """Create side control panel"""
        controls_layout = QVBoxLayout()
        
        # Tone Generator Controls
        self.tone_group = QGroupBox("Tone Generator")
        tone_layout = QVBoxLayout()
        
        # Left channel
        left_group = QGroupBox("Left Channel")
        left_layout = QVBoxLayout()
        
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave:"))
        self.left_waveform = QComboBox()
        self.left_waveform.addItems(["sine", "square", "triangle", "sawtooth"])
        self.left_waveform.currentTextChanged.connect(lambda text: setattr(self.audio, 'left_wave', text))
        wave_layout.addWidget(self.left_waveform)
        left_layout.addLayout(wave_layout)
        
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency:"))
        self.left_freq_spin = QDoubleSpinBox()
        self.left_freq_spin.setDecimals(3)
        self.left_freq_spin.setSingleStep(0.001)
        self.left_freq_spin.setRange(20.0, 2000.0)
        self.left_freq_spin.setValue(440.0)
        self.left_freq_spin.valueChanged.connect(lambda val: setattr(self.audio, 'left_freq', val))
        freq_layout.addWidget(self.left_freq_spin)
        freq_layout.addWidget(QLabel("Hz"))
        left_layout.addLayout(freq_layout)
        
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Volume:"))
        self.left_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.left_vol_slider.setRange(0, 100)
        self.left_vol_slider.setValue(20)
        self.left_vol_slider.valueChanged.connect(lambda val: setattr(self.audio, 'left_amp', val / 100))
        vol_layout.addWidget(self.left_vol_slider)
        left_layout.addLayout(vol_layout)
        
        left_group.setLayout(left_layout)
        tone_layout.addWidget(left_group)
        
        # Right channel
        right_group = QGroupBox("Right Channel")
        right_layout = QVBoxLayout()
        
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave:"))
        self.right_waveform = QComboBox()
        self.right_waveform.addItems(["sine", "square", "triangle", "sawtooth"])
        self.right_waveform.currentTextChanged.connect(lambda text: setattr(self.audio, 'right_wave', text))
        wave_layout.addWidget(self.right_waveform)
        right_layout.addLayout(wave_layout)
        
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency:"))
        self.right_freq_spin = QDoubleSpinBox()
        self.right_freq_spin.setDecimals(3)
        self.right_freq_spin.setSingleStep(0.001)
        self.right_freq_spin.setRange(20.0, 2000.0)
        self.right_freq_spin.setValue(440.0)
        self.right_freq_spin.valueChanged.connect(lambda val: setattr(self.audio, 'right_freq', val))
        freq_layout.addWidget(self.right_freq_spin)
        freq_layout.addWidget(QLabel("Hz"))
        right_layout.addLayout(freq_layout)
        
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Volume:"))
        self.right_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.right_vol_slider.setRange(0, 100)
        self.right_vol_slider.setValue(20)
        self.right_vol_slider.valueChanged.connect(lambda val: setattr(self.audio, 'right_amp', val / 100))
        vol_layout.addWidget(self.right_vol_slider)
        right_layout.addLayout(vol_layout)
        
        right_group.setLayout(right_layout)
        tone_layout.addWidget(right_group)
        
        self.tone_group.setLayout(tone_layout)
        controls_layout.addWidget(self.tone_group)
        
        # Display Controls
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout()
        
        # Invert Y
        self.invert_y_checkbox = QCheckBox("Invert Y Axis")
        self.invert_y_checkbox.toggled.connect(lambda checked: setattr(self.scope, 'invert_y', checked))
        display_layout.addWidget(self.invert_y_checkbox)
        
        # Phosphor decay
        self.phosphor_checkbox = QCheckBox("Phosphor Decay")
        self.phosphor_checkbox.setChecked(True)
        self.phosphor_checkbox.toggled.connect(lambda checked: setattr(self.scope, 'phosphor_decay', checked))
        display_layout.addWidget(self.phosphor_checkbox)
        
        # Scale controls
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 100)
        self.scale_slider.setValue(45)
        self.scale_slider.valueChanged.connect(self.update_scale)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QLabel("45%")
        scale_layout.addWidget(self.scale_label)
        display_layout.addLayout(scale_layout)
        
        # Trail
        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail:"))
        self.trail_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_slider.setRange(0, 255)
        self.trail_slider.setValue(180)
        self.trail_slider.valueChanged.connect(lambda val: setattr(self.scope, 'trail_alpha', val))
        trail_layout.addWidget(self.trail_slider)
        display_layout.addLayout(trail_layout)

        # Beam width
        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel("Beam:"))
        self.beam_slider = QSlider(Qt.Orientation.Horizontal)
        self.beam_slider.setRange(1, 10)
        self.beam_slider.setValue(2)
        self.beam_slider.valueChanged.connect(lambda val: setattr(self.scope, 'beam_width', val))
        beam_layout.addWidget(self.beam_slider)
        display_layout.addLayout(beam_layout)
        
        # Glow
        glow_layout = QHBoxLayout()
        glow_layout.addWidget(QLabel("Glow:"))
        self.glow_slider = QSlider(Qt.Orientation.Horizontal)
        self.glow_slider.setRange(0, 255)
        self.glow_slider.setValue(150)
        self.glow_slider.valueChanged.connect(lambda val: setattr(self.scope, 'glow_intensity', val))
        glow_layout.addWidget(self.glow_slider)
        display_layout.addLayout(glow_layout)
        
        # Bloom radius
        bloom_layout = QHBoxLayout()
        bloom_layout.addWidget(QLabel("Bloom:"))
        self.bloom_slider = QSlider(Qt.Orientation.Horizontal)
        self.bloom_slider.setRange(5, 30)
        self.bloom_slider.setValue(15)
        self.bloom_slider.valueChanged.connect(lambda val: setattr(self.scope, 'bloom_radius', val))
        bloom_layout.addWidget(self.bloom_slider)
        display_layout.addLayout(bloom_layout)
        
        # Color
        hue_layout = QHBoxLayout()
        hue_layout.addWidget(QLabel("Color:"))
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 360)
        self.hue_slider.setValue(120)
        self.hue_slider.valueChanged.connect(lambda val: setattr(self.scope, 'hue', val))
        hue_layout.addWidget(self.hue_slider)
        display_layout.addLayout(hue_layout)
        
        display_group.setLayout(display_layout)
        controls_layout.addWidget(display_group)
        
        controls_layout.addStretch()
        parent_layout.addLayout(controls_layout, 1)

    def update_scale(self, value):
        self.scope.x_scale = value / 100.0
        self.scope.y_scale = value / 100.0
        self.scale_label.setText(f"{value}%")

    def switch_audio_mode(self, mode):
        is_file_mode = mode == "Audio File"
        self.load_button.setEnabled(is_file_mode)
        self.tone_group.setEnabled(not is_file_mode)
        
        if not is_file_mode:
            self.audio.is_file_mode = False
            self.audio.generate_continuous_buffer()
            self.file_label.setText("No file loaded")
            self.statusBar().showMessage("Switched to Tone Generator mode")

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", 
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.statusBar().showMessage("Loading audio file...")
            self.load_button.setEnabled(False)
            
            # Start loading in thread
            self.loader_thread = AudioLoaderThread(file_path)
            self.loader_thread.loaded.connect(self.on_audio_loaded)
            self.loader_thread.error.connect(self.on_audio_error)
            self.loader_thread.start()

    def on_audio_loaded(self, audio_info, file_path):
        self.audio.set_file_audio(audio_info, file_path)
        self.file_label.setText(os.path.basename(file_path))
        self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")
        self.load_button.setEnabled(True)

    def on_audio_error(self, error_msg):
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.load_button.setEnabled(True)
        print(f"Audio loading error: {error_msg}")

    def update_scope(self):
        if self.audio.latest_stereo is None:
            return
        
        buffer = self.audio.latest_stereo
        w = self.scope.width()
        h = self.scope.height()
        
        if w <= 0 or h <= 0:
            return
        
        # Create image with alpha channel for effects
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor(0, 0, 0, 255))
        
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x, center_y = w // 2, h // 2
        x_scale = min(w, h) * self.scope.x_scale
        y_scale = min(w, h) * self.scope.y_scale
        
        # Get display window
        if self.audio.is_file_mode:
            window_size = 2048
            t = time.time()
            elapsed = t - self.audio.file_start_time if self.audio.file_start_time else 0
            sample_offset = int((elapsed * 44100) % max(1, len(buffer) - window_size))
        else:
            window_size = 2048
            t = time.time()
            sample_offset = int((t * 44100) % (len(buffer) - window_size))
        
        data = buffer[sample_offset:sample_offset + window_size]
        display_step = max(1, len(data) // 512)
        data = data[::display_step]
        
        # Convert to screen coordinates
        path = []
        for point in data:
            x = float(center_x + point[0] * x_scale)
            y = float(center_y + point[1] * y_scale if self.scope.invert_y else center_y - point[1] * y_scale)
            path.append(QPointF(x, y))
        
        # Update path history
        self.scope.path_history.extend(path)
        if len(self.scope.path_history) > self.scope.max_path_length:
            self.scope.path_history = self.scope.path_history[-self.scope.max_path_length:]
        
        # Apply shader effects
        if self.scope.glow_intensity > 0:
            ShaderEffects.apply_bloom(painter, path, self.scope.hue, self.scope.glow_intensity, self.scope.bloom_radius)
        
        # Draw phosphor trail
        if self.scope.phosphor_decay:
            ShaderEffects.draw_phosphor_trail(painter, self.scope.path_history, self.scope.hue, self.scope.trail_alpha, self.scope.beam_width)
        else:
            # Simple trail
            pen = QPen(QColor.fromHsv(self.scope.hue, 255, 255, self.scope.trail_alpha))
            pen.setWidthF(self.scope.beam_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            for i in range(1, len(path)):
                painter.drawLine(path[i - 1], path[i])

        # Optionally draw a bright core
        ShaderEffects.draw_beam_core(painter, path, self.scope.glow_intensity)

        painter.end()
        self.scope.setPixmap(QPixmap.fromImage(img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    pygame.quit()
