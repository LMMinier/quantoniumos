# QuantSoundDesign - Technical Documentation

## Overview

**QuantSoundDesign** is an in-development sound design studio prototype built on QuantoniumOS with native Φ-RFT (Recursive Fibonacci Transform) integration. It provides an FL Studio/Ableton-inspired workflow for music production, audio synthesis, and pattern-based composition.

> **STATUS: TESTING / IN DEVELOPMENT**  
> QuantSoundDesign is functional but still under active development. Features may change.

## Architecture

```
src/apps/quantsounddesign/
├── gui.py            # Main UI (3231 lines) - PyQt5 main window, views, styles
├── engine.py         # Core Engine (895 lines) - Session, tracks, clips, audio graph
├── synth_engine.py   # Synthesizer (731 lines) - PolySynth, RFT oscillators
├── pattern_editor.py # Pattern System (1651 lines) - Step sequencer, drums
├── audio_backend.py  # Audio I/O (454 lines) - Real-time audio via sounddevice
├── piano_roll.py     # MIDI Editor - Note editing and piano roll
├── devices.py        # Audio Devices - Effects, instruments
└── __init__.py
```

## Component Deep Dive

### 1. GUI Layer (`gui.py`)

The main PyQt5 interface providing:

- **Main Window** - Professional dark theme with gradient aesthetics
- **Transport Bar** - Play, pause, stop, record, tempo, time signature
- **Track View** - Arrangement view with horizontal track lanes
- **Mixer View** - Channel strips with faders, sends, FX
- **Pattern View** - Step sequencer grid
- **Instrument Browser** - Preset library and sound selection

**Key Classes:**
- `QuantSoundDesign(QMainWindow)` - Main application window
- `TransportWidget` - Playback controls
- `TrackLane` - Individual track visualization
- `MixerStrip` - Per-channel mixer controls
- `PatternGrid` - Step sequencer grid widget

### 2. Core Engine (`engine.py`)

The audio processing backend with UnitaryRFT integration:

```python
# UnitaryRFT is connected for native Φ-RFT processing
from algorithms.rft.kernels.python_bindings.unitary_rft import (
    UnitaryRFT,
    RFT_VARIANT_HARMONIC,
    RFT_VARIANT_FIBONACCI,
    # ... 7 total variants
)
```

**Key Classes:**
- `Session` - Complete project state (tracks, tempo, patterns)
- `AudioEngine` - Real-time audio graph processing
- `Track` - Audio/MIDI track container
- `Clip` - Audio or MIDI data unit
- `WaveField` - Core signal abstraction (time or RFT domain)
- `Device` - Effect or instrument in chain

**RFT Variants Available:**
| Variant | Use Case |
|---------|----------|
| `STANDARD` | General purpose |
| `HARMONIC` | Harmonic series analysis |
| `FIBONACCI` | Fibonacci-spaced processing |
| `CHAOTIC` | Non-linear dynamics |
| `GEOMETRIC` | Geometric sequences |
| `HYBRID` | Combined approaches |
| `ADAPTIVE` | Data-dependent selection |

## Running QuantSoundDesign

### Prerequisites

```bash
pip install PyQt5 numpy sounddevice
```

### Launch

```python
from src.apps.quantsounddesign.gui import QuantSoundDesign
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
qsd = QuantSoundDesign()
qsd.show()
sys.exit(app.exec_())
```

Or via the QuantoniumOS desktop:
```python
python quantonium_os_src/frontend/quantonium_desktop.py
# Click "QuantSoundDesign" in the launcher
```

## License

See `LICENSE.md` and `LICENSE-CLAIMS-NC.md` in the root directory.

---

**QuantSoundDesign** - Φ-RFT Sound Design Studio  
*Part of QuantoniumOS - Building the future of audio with golden ratio mathematics.*

### 2. Core Engine (`engine.py`)

The audio processing backend with UnitaryRFT integration:

```python
# UnitaryRFT is connected for native Φ-RFT processing
from algorithms.rft.kernels.python_bindings.unitary_rft import (
    UnitaryRFT,
    RFT_VARIANT_HARMONIC,
    RFT_VARIANT_FIBONACCI,
    # ... 7 total variants
)
```

**Key Classes:**
- `Session` - Complete project state (tracks, tempo, patterns)
- `AudioEngine` - Real-time audio graph processing
- `Track` - Audio/MIDI track container
- `Clip` - Audio or MIDI data unit
- `WaveField` - Core signal abstraction (time or RFT domain)
- `Device` - Effect or instrument in chain

**RFT Variants Available:**
| Variant | Use Case |
|---------|----------|
| `STANDARD` | General purpose |
| `HARMONIC` | Harmonic series analysis |
| `FIBONACCI` | Fibonacci-spaced processing |
| `CHAOTIC` | Non-linear dynamics |
| `GEOMETRIC` | Geometric sequences |
| `HYBRID` | Combined approaches |
| `ADAPTIVE` | Data-dependent selection |

### 3. Synthesizer Engine (`synth_engine.py`)

Polyphonic synthesizer with RFT-based wave shaping:

```python
# RFT Additive Synthesis
def rft_additive_synthesis(freq, duration, sample_rate, num_harmonics=8):
    """
    Generate waveform using UnitaryRFT additive synthesis.
    Creates harmonics in RFT domain and transforms back.
    - Places harmonics at Φ-scaled positions
    - Uses UnitaryRFT.inverse() for time-domain conversion
    """
```

**Features:**
- 8-voice polyphony
- Multiple oscillator types (sine, saw, square, triangle, RFT)
- ADSR envelope
- LFO modulation
- Computer keyboard input (ASDFGHJK = piano keys)

**Key Classes:**
- `PolySynth` - Main polyphonic synthesizer
- `Voice` - Individual voice with oscillator + envelope
- `Oscillator` - Waveform generator
- `ADSR` - Envelope generator

### 4. Pattern Editor (`pattern_editor.py`)

Step sequencer with Φ-RFT enhanced drum synthesis:

```python
# RFTMW integration for Φ-enhanced synthesis
from quantonium_os_src.engine.RFTMW import MiddlewareTransformEngine

_rft_engine = MiddlewareTransformEngine()
```

**Drum Types (16 total):**
- KICK, SNARE, CLAP
- HIHAT_CLOSED, HIHAT_OPEN
- TOM_HIGH, TOM_MID, TOM_LOW
- CRASH, RIDE, RIMSHOT
- COWBELL, SHAKER, CLAV
- PERC_1, PERC_2

**Key Classes:**
- `Pattern` - Pattern container with rows and steps
- `PatternRow` - Single drum/instrument lane
- `PatternStep` - Individual step with velocity/gate
- `PatternEditorWidget` - PyQt5 grid UI
- `DrumSynthesizer` - Φ-RFT enhanced drum synthesis
- `PatternPlayer` - Real-time pattern playback

**Step Preview Feature:**
When toggling a step, the drum sound plays immediately for instant feedback.

### 5. Audio Backend (`audio_backend.py`)

Real-time audio I/O using `sounddevice`:

```python
import sounddevice as sd

class AudioBackend:
    """
    Real-time audio backend using sounddevice.
    
    Integrates with Session/AudioEngine for:
    - Playing back DAW tracks through the audio graph
    - Synth input for real-time keyboard playing
    - Metronome and click track
    """
```

**Configuration:**
- Sample rate: 44100 Hz (default)
- Block size: 512 samples
- Channels: 2 (stereo)
- Format: float32

**Features:**
- Low-latency audio output
- Preview sound system (one-shots)
- Metronome/click track
- Thread-safe audio callback

## RFT Integration Points

### 1. Engine → UnitaryRFT

The core engine uses the native UnitaryRFT library for:
- Transform domain processing
- Φ-scaled spectral analysis
- Waveform domain representation

### 2. Synth → UnitaryRFT

The synthesizer uses RFT for:
- Additive synthesis with Φ-spaced harmonics
- Wave shaping in RFT domain
- Timbre generation (280x coverage vs standard oscillators)

### 3. Pattern → RFTMW

The pattern editor uses the middleware engine for:
- Φ-enhanced drum synthesis
- Spectral shaping of percussion
- Real-time transform processing

## Data Flow

```
User Input (keyboard/mouse)
         │
         ▼
    ┌─────────┐
    │  GUI    │ ◄─── gui.py
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Session │ ◄─── engine.py (tracks, clips, patterns)
    └────┬────┘
         │
         ▼
    ┌────────────┐
    │AudioEngine │ ◄─── engine.py (audio graph processing)
    └────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌─────────────┐
│ Synth │ │PatternPlayer│ ◄─── synth_engine.py, pattern_editor.py
└───┬───┘ └──────┬──────┘
    │            │
    ▼            ▼
┌──────────────────┐
│  Audio Backend   │ ◄─── audio_backend.py (sounddevice output)
└──────────────────┘
         │
         ▼
    [Speakers/DAC]
```

## Running Wave DAW

### Prerequisites

```bash
pip install PyQt5 numpy sounddevice
```

### Launch

```python
from src.apps.wave_daw.gui import WaveDAWPro
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
daw = WaveDAWPro()
daw.show()
sys.exit(app.exec_())
```

Or via the QuantoniumOS desktop:
```python
python src/quantonium_desktop.py
# Click "Wave DAW" in the launcher
```

## Current Features (v0.x Testing)

✅ **Working:**
- Main window with professional dark theme
- Transport controls (play/pause/stop)
- Track lanes (audio + instrument)
- Step sequencer with 16 steps
- Drum synthesis (16 drum types)
- Polyphonic synth with keyboard input
- Pattern editor with step preview
- Blank session startup (no demo data)
- Splitter handles for view resizing

🔄 **In Progress:**
- Audio recording
- MIDI import/export
- Effect chains
- Automation lanes
- Project save/load
- Stem export (individual track WAV files)

❌ **Not Yet Implemented:**
- Audio clip editing
- Time stretching
- Sidechaining
- Plugin hosting (VST/AU)

## File Formats

### Project Format (Planned)
```json
{
  "version": "1.0",
  "tempo": 120,
  "time_signature": [4, 4],
  "tracks": [...],
  "patterns": [...],
  "mixer_state": {...}
}
```

## Contributing

Wave DAW is part of the QuantoniumOS project. When contributing:

1. Follow the existing code style
2. Test with both GUI and headless modes
3. Document any new RFT integration points
4. Update this README for new features

## Dependencies

| Package | Purpose |
|---------|---------|
| PyQt5 | GUI framework |
| numpy | Numerical processing |
| sounddevice | Audio I/O |
| UnitaryRFT | Native Φ-RFT transforms |
| RFTMW | Middleware transform engine |

## License

See `LICENSE.md` and `LICENSE-CLAIMS-NC.md` in the root directory.

---

**Wave DAW Pro** - Φ-RFT Native Digital Audio Workstation  
*Part of QuantoniumOS - Building the future of audio with golden ratio mathematics.*
