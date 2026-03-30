import type { Metadata } from 'next';
import './globals.css';

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
      <body>{children}</body>
    </html>
  );
}
