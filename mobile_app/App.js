import React, { useEffect, useState } from 'react';
import { SafeAreaView, StatusBar, StyleSheet, Text, View, Alert, Platform } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import messaging from '@react-native-firebase/messaging';
import PushNotification from 'react-native-push-notification';

// Import screens (these would be created in separate files)
const HomeScreen = ({ navigation }) => {
  return (
    <View style={styles.screenContainer}>
      <Text style={styles.title}>Fall Detection Monitor</Text>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>System Status</Text>
        <Text style={styles.statusText}>Monitoring Active</Text>
      </View>
      <View style={styles.buttonContainer}>
        <Text style={styles.button} onPress={() => navigation.navigate('Events')}>View Events</Text>
        <Text style={styles.button} onPress={() => navigation.navigate('Settings')}>Settings</Text>
      </View>
    </View>
  );
};

const EventsScreen = () => {
  const [events, setEvents] = useState([]);
  
  useEffect(() => {
    // Fetch events from API
    fetchEvents();
  }, []);
  
  const fetchEvents = async () => {
    try {
      // Replace with your actual API endpoint
      const response = await fetch('http://your-server-ip:5000/api/events');
      const data = await response.json();
      setEvents(data);
    } catch (error) {
      console.error('Error fetching events:', error);
    }
  };
  
  return (
    <View style={styles.screenContainer}>
      <Text style={styles.title}>Fall Detection Events</Text>
      {events.length === 0 ? (
        <Text style={styles.emptyText}>No events recorded</Text>
      ) : (
        events.map(event => (
          <View key={event.id} style={styles.eventCard}>
            <Text style={styles.eventTime}>{new Date(event.timestamp).toLocaleString()}</Text>
            <Text style={styles.eventConfidence}>Confidence: {event.confidence.toFixed(2)}</Text>
            {event.snapshot_url && (
              <Text style={styles.viewSnapshot} onPress={() => navigation.navigate('ViewSnapshot', { url: event.snapshot_url })}>View Snapshot</Text>
            )}
          </View>
        ))
      )}
    </View>
  );
};

const SettingsScreen = () => {
  return (
    <View style={styles.screenContainer}>
      <Text style={styles.title}>Settings</Text>
      <Text style={styles.settingLabel}>Notification Settings</Text>
      {/* Settings would go here */}
    </View>
  );
};

const ViewSnapshotScreen = ({ route }) => {
  const { url } = route.params;
  
  return (
    <View style={styles.screenContainer}>
      <Text style={styles.title}>Fall Detection Snapshot</Text>
      {/* Image component would go here */}
      <Text>Image URL: {url}</Text>
    </View>
  );
};

// Create the navigation stack
const Stack = createStackNavigator();

// Configure push notifications
const configurePushNotifications = () => {
  PushNotification.configure({
    onRegister: function(token) {
      console.log('TOKEN:', token);
      // Send token to server
      registerDeviceWithServer(token);
    },
    onNotification: function(notification) {
      console.log('NOTIFICATION:', notification);
      // Handle notification
      if (notification.foreground) {
        Alert.alert(
          'Fall Detected',
          'A fall has been detected. Check the app for details.',
          [{ text: 'OK' }]
        );
      }
    },
    permissions: {
      alert: true,
      badge: true,
      sound: true,
    },
    popInitialNotification: true,
    requestPermissions: true,
  });
};

// Register device with server
const registerDeviceWithServer = async (token) => {
  try {
    // Replace with your actual API endpoint
    const response = await fetch('http://your-server-ip:5000/api/register-device', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        token: token.token,
        platform: Platform.OS,
      }),
    });
    const data = await response.json();
    console.log('Device registered:', data);
  } catch (error) {
    console.error('Error registering device:', error);
  }
};

// Request Firebase messaging permissions
const requestUserPermission = async () => {
  const authStatus = await messaging().requestPermission();
  const enabled =
    authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
    authStatus === messaging.AuthorizationStatus.PROVISIONAL;

  if (enabled) {
    console.log('Authorization status:', authStatus);
    // Get FCM token
    const token = await messaging().getToken();
    console.log('FCM Token:', token);
    // Register with server
    registerDeviceWithServer({ token });
  }
};

// Main App component
function App() {
  useEffect(() => {
    // Configure push notifications
    configurePushNotifications();
    
    // Request Firebase messaging permissions
    requestUserPermission();
    
    // Set up Firebase message listener
    const unsubscribe = messaging().onMessage(async remoteMessage => {
      console.log('A new FCM message arrived!', JSON.stringify(remoteMessage));
      Alert.alert(
        remoteMessage.notification.title,
        remoteMessage.notification.body,
        [{ text: 'OK' }]
      );
    });

    return unsubscribe;
  }, []);

  return (
    <NavigationContainer>
      <StatusBar barStyle="dark-content" />
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'Fall Detection Monitor' }} />
        <Stack.Screen name="Events" component={EventsScreen} options={{ title: 'Fall Events' }} />
        <Stack.Screen name="Settings" component={SettingsScreen} options={{ title: 'Settings' }} />
        <Stack.Screen name="ViewSnapshot" component={ViewSnapshotScreen} options={{ title: 'Snapshot' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  screenContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  statusText: {
    fontSize: 16,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    backgroundColor: '#2196F3',
    color: '#fff',
    padding: 15,
    borderRadius: 5,
    textAlign: 'center',
    fontWeight: 'bold',
    width: '48%',
  },
  eventCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 1,
  },
  eventTime: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  eventConfidence: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  viewSnapshot: {
    color: '#2196F3',
    marginTop: 10,
    fontWeight: 'bold',
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 30,
  },
  settingLabel: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
});

export default App;