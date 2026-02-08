/**
 * App Navigator - QuantoniumOS Mobile
 * Main navigation structure
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';

// Import screens
import LauncherScreen from '../screens/LauncherScreen';
import AIChatScreen from '../screens/AIChatScreen';
import StructuralHealthScreen from '../screens/StructuralHealthScreen';
import ChipViewer3DScreen from '../screens/ChipViewer3DScreen';
import QNotesScreen from '../screens/QNotesScreen';
import QVaultScreen from '../screens/QVaultScreen';
import QuantumCryptographyScreen from '../screens/QuantumCryptographyScreen';
import QuantumSimulatorScreen from '../screens/QuantumSimulatorScreen';
import RFTVisualizerScreen from '../screens/RFTVisualizerScreen';
import SystemMonitorScreen from '../screens/SystemMonitorScreen';
import ValidationScreen from '../screens/ValidationScreen';

export type RootStackParamList = {
  Launcher: undefined;
  AIChat: undefined;
  StructuralHealth: undefined;
  ChipViewer3D: undefined;
  QNotes: undefined;
  QVault: undefined;
  QuantumCryptography: undefined;
  QuantumSimulator: undefined;
  RFTVisualizer: undefined;
  SystemMonitor: undefined;
  Validation: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Launcher"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#1a1a2e',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
            contentStyle: {
              backgroundColor: '#16213e',
            },
          }}
        >
          <Stack.Screen
            name="Launcher"
            component={LauncherScreen}
            options={{
              title: 'QuantoniumOS',
              headerShown: false, // Fullscreen like desktop
            }}
          />
          <Stack.Screen
            name="AIChat"
            component={AIChatScreen}
            options={{ title: 'AI Chat' }}
          />
          <Stack.Screen
            name="StructuralHealth"
            component={StructuralHealthScreen}
            options={{ title: 'Structural Health Monitor' }}
          />
          <Stack.Screen
            name="ChipViewer3D"
            component={ChipViewer3DScreen}
            options={{ title: 'RFTPU 3D Viewer' }}
          />
          <Stack.Screen
            name="QNotes"
            component={QNotesScreen}
            options={{ title: 'Quantum Notes' }}
          />
          <Stack.Screen
            name="QVault"
            component={QVaultScreen}
            options={{ title: 'Quantum Vault' }}
          />
          <Stack.Screen
            name="QuantumCryptography"
            component={QuantumCryptographyScreen}
            options={{ title: 'Quantum Cryptography' }}
          />
          <Stack.Screen
            name="QuantumSimulator"
            component={QuantumSimulatorScreen}
            options={{ title: 'Quantum Simulator' }}
          />
          <Stack.Screen
            name="RFTVisualizer"
            component={RFTVisualizerScreen}
            options={{ title: 'RFT Visualizer' }}
          />
          <Stack.Screen
            name="SystemMonitor"
            component={SystemMonitorScreen}
            options={{ title: 'System Monitor' }}
          />
          <Stack.Screen
            name="Validation"
            component={ValidationScreen}
            options={{ title: 'Validation Suite' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}
