# Structural Health Monitoring (SHM) Engine

## Overview

Research-grade vibration analysis prototype using **Φ-RFT (Golden-Ratio Resonant Fourier Transform)** for structural health monitoring across five specialized domains.

**Disclaimer:** Standards referenced below are for contextual mapping only. This prototype is not certified, calibrated, or validated for regulatory or safety-critical use.

## Domains

### 🌍 Seismic Monitoring
**Application:** Earthquake detection, building earthquake response

| Function | Description | Standards |
|----------|-------------|-----------|
| `analyzeSeismic()` | Main seismic analysis | - |
| `computePGA()` | Peak Ground Acceleration | Eurocode 8 |
| `computePGV()` | Peak Ground Velocity | - |
| `computePGD()` | Peak Ground Displacement | - |
| `computeArias()` | Arias Intensity (m/s) | Arias 1970 |
| `computeCAV()` | Cumulative Absolute Velocity | NRC RG 1.166 |
| `getMercalliIntensity()` | Modified Mercalli scale | USGS |
| `estimateMagnitude()` | Local magnitude estimate | Gutenberg-Richter |
| `getSignificantDuration()` | 5-95% Arias duration | Trifunac & Brady |
| `getDominantFrequency()` | Peak spectral frequency | - |

**Alerts:**
- Strong shaking (PGA > 0.1g)
- Severe shaking (PGA > 0.3g)
- High CAV (> 0.16 m/s)
- Mercalli intensity reporting

---

### 🌉 Bridge Health Monitoring
**Application:** Traffic vibration, structural fatigue, cable tension

| Function | Description | Standards |
|----------|-------------|-----------|
| `analyzeBridge()` | Main bridge analysis | - |
| `getDeckFrequency()` | First vertical mode | - |
| `getTorsionalFrequency()` | First torsional mode | - |
| `computeDamping()` | Modal damping ratio | Half-power method |
| `getFrequencyShift()` | % change from baseline | - |
| `getDeckAcceleration()` | RMS deck motion | - |
| `getTrafficIndex()` | Traffic intensity (0-1) | - |
| `getWindResponse()` | Wind-induced vibration | - |
| `getCableForceIndex()` | Cable tension indicator | - |
| `computeHealthIndex()` | Overall health (0-100) | - |
| `checkSpectralCoherence()` | Baseline comparison | - |

**Alerts:**
- Natural frequency shift > 5%
- Critical frequency shift > 15%
- High damping ratio > 10%
- Excessive deck acceleration > 1.0 m/s²
- Low spectral coherence < 0.8

---

### 🏢 Building Sway Analysis
**Application:** Wind-induced motion, occupant comfort, tall building monitoring

| Function | Description | Standards |
|----------|-------------|-----------|
| `analyzeBuilding()` | Main building analysis | - |
| `getSwayFrequency()` | Fundamental frequency | - |
| `getSwayAmplitude()` | Displacement (mm) | Double integration |
| `getSwayDirection()` | Direction from X/Y | - |
| `getDampingRatio()` | Structural damping | Half-power method |
| `getAccelerationRMS()` | RMS acceleration | - |
| `getPeakAcceleration()` | Maximum acceleration | - |
| `getComfortLevel()` | Occupant comfort | ISO 10137 |
| `getHumanPerception()` | Perceptibility assessment | ISO 2631 |
| `estimateWindSpeed()` | Wind speed from response | - |
| `getTorsionIndex()` | Torsional/translational ratio | - |

**Comfort Levels (ISO 10137):**
- Comfortable: < 0.05 m/s²
- Perceptible: 0.05-0.1 m/s²
- Annoying: 0.1-0.25 m/s²
- Uncomfortable: 0.25-0.5 m/s²
- Unacceptable: > 0.5 m/s²

**Human Perception (ISO 2631):**
- Imperceptible: < 0.01 m/s²
- Barely perceptible: 0.01-0.04 m/s²
- Perceptible: 0.04-0.125 m/s²
- Annoying: 0.125-0.4 m/s²
- Uncomfortable: 0.4-1.25 m/s²
- Harmful: > 4.0 m/s²

---

### 🏗️ Construction Compliance
**Application:** Pile driving, excavation, ground-borne vibration

| Function | Description | Standards |
|----------|-------------|-----------|
| `analyzeConstruction()` | Main construction analysis | - |
| `computePPV()` | Peak Particle Velocity (mm/s) | DIN 4150-3 |
| `getDominantFrequency()` | Vibration frequency | - |
| `checkDIN4150Compliance()` | Structural vibration limits | DIN 4150-3 |
| `checkBS7385Compliance()` | Building vibration limits | BS 7385-2 |
| `getComplianceMargin()` | % below limit | - |
| `getCrestFactor()` | Peak/RMS ratio | - |
| `computeVDV()` | Vibration Dose Value | BS 6472 |
| `estimateSourceType()` | Source classification | - |
| `getEventDuration()` | Event length (s) | - |

**DIN 4150-3 PPV Limits (mm/s):**

| Building Type | Short-term | Long-term |
|---------------|------------|-----------|
| Residential | 5 | 2.5 |
| Commercial | 15 | 5 |
| Industrial | 20 | 10 |
| Historic | 3 | 1.5 |

**Source Type Detection:**
- Impact pile driving (f < 5 Hz, CF > 5)
- Vibratory compaction (f < 10 Hz, CF < 3)
- Construction machinery (20-100 Hz)
- High-speed equipment (> 100 Hz)

---

### ⚙️ Industrial Machine Health
**Application:** Rotating machinery, bearings, pumps, motors

| Function | Description | Standards |
|----------|-------------|-----------|
| `analyzeMachine()` | Main machine analysis | - |
| `estimateRPM()` | Rotation speed | From 1X frequency |
| `getFundamentalFrequency()` | 1X frequency (Hz) | - |
| `extractHarmonics()` | 2X, 3X, 4X... amplitudes | - |
| `getOverallVibration()` | RMS velocity (mm/s) | ISO 10816 |
| `getUnbalanceIndex()` | 1X amplitude indicator | - |
| `getMisalignmentIndex()` | 2X amplitude indicator | - |
| `getBearingCondition()` | ISO 10816 assessment | ISO 10816 |
| `getLoosenesIndex()` | Sub-harmonic indicator | - |
| `getCavitationIndex()` | High-freq noise | - |
| `computeHealthScore()` | Overall health (0-100) | - |
| `getRemainingLife()` | Life estimate | - |

**ISO 10816 Vibration Limits (mm/s RMS):**

| Zone | Class I | Description |
|------|---------|-------------|
| A | < 0.71 | Good |
| B | 0.71-1.8 | Acceptable |
| C | 1.8-4.5 | Unsatisfactory |
| D | > 4.5 | Unacceptable |

**Fault Detection:**
- Unbalance: High 1X amplitude
- Misalignment: High 2X amplitude (> 50% of 1X)
- Mechanical looseness: Sub-harmonic presence
- Bearing wear: High-frequency noise
- Cavitation: Broad high-frequency energy

**Remaining Life Assessment:**
- Normal: Health > 80
- Monitor closely: 60-80
- Schedule maintenance: 40-60
- Urgent maintenance: 20-40
- Immediate action: < 20

---

## Signal Processing Functions

### Core Analysis
| Function | Description |
|----------|-------------|
| `preprocess()` | DC removal + Hanning window |
| `computeFFT()` | Cooley-Tukey FFT |
| `computeRFT()` | Φ-RFT with golden-ratio phase |
| `integrate()` | Time integration with drift removal |
| `computeRMS()` | Root Mean Square |

### Spectral Analysis
| Function | Description |
|----------|-------------|
| `computeSpectralAnalysis()` | Full spectral decomposition |
| `computePSD()` | Power Spectral Density |
| `findPeaks()` | Peak frequency detection |
| `computeBandEnergies()` | Energy in frequency bands |
| `computeSpectralCentroid()` | Frequency center of mass |
| `computeSpectralRolloff()` | 85% energy frequency |
| `computeSpectralEntropy()` | Spectral flatness |

### Modal Analysis
| Function | Description |
|----------|-------------|
| `extractModes()` | Modal parameter extraction |
| `estimateDamping()` | Half-power bandwidth method |

### RFT-Specific
| Function | Description |
|----------|-------------|
| `computeRFT()` | Golden-ratio phase modulation |
| `computeSparsity()` | RFT coefficient sparsity |
| `computeCoherence()` | Spectral coherence |

---

## Frequency Bands

| Band | Range | Application |
|------|-------|-------------|
| Very Low | 0.1-2 Hz | Building sway, wind |
| Low | 2-10 Hz | Structural modes |
| Mid | 10-50 Hz | Machinery, traffic |
| High | 50-200 Hz | Bearings, equipment |

---

## Usage Example

```typescript
import { SHMEngine, SHMDomain } from './SHMEngine';

// Initialize for bridge monitoring at 100 Hz
const engine = new SHMEngine('bridge', 100);

// Collect samples from accelerometer
const samples = collectAccelerometerData(); // AccelerometerSample[]

// Set baseline (during normal operation)
engine.setBaseline(baselineSamples);

// Analyze current data
const result = engine.analyze(samples);

// Check results
console.log('Deck frequency:', result.bridge?.deckFrequency, 'Hz');
console.log('Health index:', result.bridge?.healthIndex);
console.log('Alerts:', result.alerts);
```

---

## Engineering Standards Referenced

- **ISO 4866** - Mechanical vibration guidelines
- **ISO 10816** - Machine vibration evaluation
- **ISO 10137** - Serviceability of buildings (occupant comfort)
- **ISO 2631** - Human exposure to vibration
- **DIN 4150-3** - Structural vibration in buildings
- **BS 7385-2** - Evaluation of building vibrations
- **BS 6472** - Human exposure to vibration in buildings
- **Eurocode 8** - Design for earthquake resistance
- **NRC RG 1.166** - Pre-earthquake planning (CAV)

---

## Typical Structural Frequencies

| Structure | Frequency Range |
|-----------|-----------------|
| Tall building (50+ stories) | 0.1-0.5 Hz |
| Mid-rise (10-30 stories) | 0.5-2.0 Hz |
| Low-rise (1-5 stories) | 2.0-5.0 Hz |
| Bridge deck | 0.5-3.0 Hz |
| Bridge cable | 0.5-10.0 Hz |
| Machinery | 10-1000 Hz |

---

## License

SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
Copyright (C) 2025 Luis M. Minier / quantoniumos
