import type { Metadata } from 'next';
import './globals.css';
import { AuthProvider } from '@/lib/auth-context';

export const metadata: Metadata = {
  title: 'Seat Occupancy Monitor — Real-Time AI Detection',
  description:
    'Real-time seat occupancy detection using YOLOv8 AI running entirely in your browser. No server, no cloud — just your camera and AI.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Load OpenCV.js for camera stabilization and advanced image processing */}
        {/* Added crossOrigin="anonymous" to resolve the runtime cross-origin error */}
        <script 
          async 
          src="https://docs.opencv.org/4.x/opencv.js" 
          type="text/javascript" 
          crossOrigin="anonymous"
        ></script>
      </head>
      <body>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
