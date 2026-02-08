/**
 * Design System Constants - QuantoniumOS Mobile
 * Golden Ratio-based design system matching desktop UI
 */

// Golden Ratio constants (matching desktop Python implementation)
export const PHI = 1.618033988749895; // φ (Golden Ratio)
export const PHI_SQ = PHI * PHI; // φ²
export const PHI_INV = 1 / PHI; // 1/φ

// Base unit for scaling (multiply by φ powers for harmony)
export const BASE_UNIT = 16;

// Spacing system based on Golden Ratio
export const spacing = {
  xs: Math.round(BASE_UNIT * PHI_INV * PHI_INV), // φ⁻²
  sm: Math.round(BASE_UNIT * PHI_INV), // φ⁻¹
  md: BASE_UNIT, // base
  lg: Math.round(BASE_UNIT * PHI), // φ
  xl: Math.round(BASE_UNIT * PHI_SQ), // φ²
  xxl: Math.round(BASE_UNIT * PHI * PHI_SQ), // φ³
};

// Typography based on Golden Ratio
export const typography = {
  micro: Math.round(BASE_UNIT * PHI_INV * 0.6),
  small: Math.round(BASE_UNIT * PHI_INV * 0.8),
  body: Math.round(BASE_UNIT * PHI_INV),
  subtitle: Math.round(BASE_UNIT),
  title: Math.round(BASE_UNIT * PHI),
  display: Math.round(BASE_UNIT * PHI_SQ),
  hero: Math.round(BASE_UNIT * PHI * PHI_SQ),
};

// Color palette matching desktop UI
export const colors = {
  // Primary colors from desktop
  primary: '#3498db', // Main blue (matching desktop Q logo)
  primaryDark: '#2980b9',
  primaryLight: '#5dade2',

  // Neutral colors from desktop
  dark: '#2c3e50', // Darkest text
  darkGray: '#34495e', // Secondary text
  gray: '#7f8c8d',
  lightGray: '#bdc3c7',
  offWhite: '#ecf0f1',
  white: '#ffffff',

  // Semantic colors
  success: '#27ae60',
  warning: '#f39c12',
  error: '#e74c3c',
  info: '#3498db',

  // Background colors
  background: '#ffffff',
  backgroundDark: '#0f0c29', // For dark mode/special screens
  surface: '#f8f9fa',
  surfaceElevated: '#ffffff',

  // App-specific gradients (kept from original for each app's unique identity)
  vaultGradient: ['#667eea', '#764ba2'],
  notesGradient: ['#f093fb', '#f5576c'],
  simulatorGradient: ['#4facfe', '#00f2fe'],
  rftGradient: ['#43e97b', '#38f9d7'],
  validationGradient: ['#fa709a', '#fee140'],
  cryptoGradient: ['#a8edea', '#fed6e3'],
  monitorGradient: ['#ff9a9e', '#fecfef'],
  aiGradient: ['#fbc2eb', '#a6c1ee'],
  chipViewerGradient: ['#667eea', '#00d4ff'],
  shmGradient: ['#FF6B6B', '#4ECDC4'],
};

// Border radius system
export const borderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
};

// Shadow elevations
export const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.18,
    shadowRadius: 1.0,
    elevation: 1,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.30,
    shadowRadius: 4.65,
    elevation: 8,
  },
};

// App metadata (matching desktop apps list)
export interface AppInfo {
  id: string;
  name: string;
  icon: string;
  description: string;
  category: string;
  screen: string;
  colors: string[];
}

export const apps: AppInfo[] = [
  {
    id: '1',
    name: 'AI Chat',
    icon: '🤖',
    description: 'Quantum-enhanced AI assistant',
    category: 'AI',
    screen: 'AIChat',
    colors: colors.aiGradient,
  },
  {
    id: '2',
    name: 'Structural Health',
    icon: '🏗️',
    description: 'Vibration-based damage detection',
    category: 'ANALYSIS',
    screen: 'StructuralHealth',
    colors: colors.shmGradient,
  },
  {
    id: '3',
    name: 'RFTPU 3D Viewer',
    icon: '🔬',
    description: '3D chip architecture visualization',
    category: 'HARDWARE',
    screen: 'ChipViewer3D',
    colors: colors.aiGradient,
  },
  {
    id: '4',
    name: 'Quantum Notes',
    icon: '📝',
    description: 'Encrypted note-taking with RFT',
    category: 'PRODUCTIVITY',
    screen: 'QNotes',
    colors: colors.shmGradient,
  },
  {
    id: '5',
    name: 'Quantum Vault',
    icon: '🔐',
    description: 'Secure file storage with RFT-SIS',
    category: 'SECURITY',
    screen: 'QVault',
    colors: colors.aiGradient,
  },
  {
    id: '6',
    name: 'Quantum Cryptography',
    icon: '🔑',
    description: 'RFT-SIS hash & Feistel cipher demo',
    category: 'SECURITY',
    screen: 'QuantumCryptography',
    colors: colors.shmGradient,
  },
  {
    id: '7',
    name: 'Quantum Simulator',
    icon: '⚛️',
    description: 'Classical quantum state simulation',
    category: 'SIMULATION',
    screen: 'QuantumSimulator',
    colors: colors.aiGradient,
  },
  {
    id: '8',
    name: 'RFT Visualizer',
    icon: '📊',
    description: 'Real-time RFT transform visualization',
    category: 'ANALYSIS',
    screen: 'RFTVisualizer',
    colors: colors.shmGradient,
  },
  {
    id: '9',
    name: 'System Monitor',
    icon: '📈',
    description: 'System performance metrics',
    category: 'SYSTEM',
    screen: 'SystemMonitor',
    colors: colors.aiGradient,
  },
  {
    id: '10',
    name: 'Validation Suite',
    icon: '✅',
    description: 'Run theorem proofs & test suites',
    category: 'TESTING',
    screen: 'Validation',
    colors: colors.shmGradient,
  },
];
