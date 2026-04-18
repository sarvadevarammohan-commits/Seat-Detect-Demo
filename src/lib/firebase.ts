import { initializeApp, getApps } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getDatabase } from 'firebase/database';

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
// Firebase configuration - uses environment variables if present
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY || "AIzaSyCbpIJ-SAI6Z1QrRBGiuu19IZY-1Mku2tk",
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || "seat-occupancy-a2662.firebaseapp.com",
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "seat-occupancy-a2662",
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET || "seat-occupancy-a2662.firebasestorage.app",
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || "307332327096",
  databaseURL: process.env.NEXT_PUBLIC_FIREBASE_DATABASE_URL || "https://seat-occupancy-a2662-default-rtdb.asia-southeast1.firebasedatabase.app/",
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID || "1:307332327096:web:99522cb9cdb1025bbe764d",
  measurementId: process.env.NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID || "G-039GFVVZS9"
};

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
const auth = getAuth(app);
const db = getDatabase(app, firebaseConfig.databaseURL);

console.log("🔥 Firebase Initialized for:", firebaseConfig.projectId);
console.log("📡 Database URL:", firebaseConfig.databaseURL);

export { app, auth, db };
