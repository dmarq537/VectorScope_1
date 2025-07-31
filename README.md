# VectorScope

This application displays a real-time vectorscope visualization. It relies on
several Python packages:

- `numpy`
- `pygame`
- `PyQt6`
- `pydub` (optional, for loading MP3/FLAC/etc.)

Install the dependencies using pip:

```bash
pip install numpy pygame PyQt6 pydub
```

If `pydub` is not installed, only uncompressed WAV files can be loaded. To start
the application run:

```bash
python Vectorscope.py
```
