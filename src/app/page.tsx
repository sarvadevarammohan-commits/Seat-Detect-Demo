'use client';

import { useState, useEffect } from 'react';
import { db } from '@/lib/firebase';
import { ref, onValue } from 'firebase/database';
import type { SeatConfig } from '@/lib/types';
import Link from 'next/link';

export default function PublicDashboard() {
  const [seats, setSeats] = useState<Record<string, SeatConfig>>({});
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const [source, setSource] = useState<'webcam' | 'file'>('file');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Listen for seats config
    const seatsRef = ref(db, 'seats');
    const unsubscribeSeats = onValue(seatsRef, (snapshot) => {
      const data = snapshot.val();
      if (data) setSeats(data);
    });

    // Listen for input source
    const sourceRef = ref(db, 'config/source');
    const unsubscribeSource = onValue(sourceRef, (snapshot) => {
      setSource(snapshot.val() || 'file');
    });

    // Listen for occupancy status
    const statusRef = ref(db, 'status');
    const unsubscribeStatus = onValue(statusRef, (snapshot) => {
      const data = snapshot.val();
      setStatus(data || {});
      setLoading(false);
    }, (error) => {
      console.error("❌ Firebase Status Sync Error:", error);
      setLoading(false);
    });

    return () => {
      unsubscribeSeats();
      unsubscribeStatus();
      unsubscribeSource();
    };
  }, []);

  const totalSeats = Object.keys(seats).length;
  const occupiedCount = Object.values(status).filter(Boolean).length;
  const freeCount = totalSeats - occupiedCount;

  // Logic to group seats into rows based on Y coordinates
  const generateDynamicLayout = () => {
    const seatList = Object.values(seats);
    if (seatList.length === 0) return [];
    
    // Sort by Y coordinate first
    const sortedByY = [...seatList].sort((a, b) => {
      const ya = a.box ? a.box[1] : (a.polygon ? Math.min(...a.polygon.map(p => p[1])) : 0);
      const yb = b.box ? b.box[1] : (b.polygon ? Math.min(...b.polygon.map(p => p[1])) : 0);
      return ya - yb;
    });
    
    const rows: SeatConfig[][] = [];
    let currentRow: SeatConfig[] = [];
    let lastY = -1;
    
    // Threshold to consider seats in the same row (10% of typical 600px height is ~60px)
    const ROW_THRESHOLD = 60;

    sortedByY.forEach(s => {
      const y = s.box ? s.box[1] : (s.polygon ? Math.min(...s.polygon.map(p => p[1])) : 0);
      if (lastY === -1 || Math.abs(y - lastY) < ROW_THRESHOLD) {
        currentRow.push(s);
      } else {
        rows.push(currentRow.sort((a, b) => {
          const xa = a.box ? a.box[0] : (a.polygon ? Math.min(...a.polygon.map(p => p[0])) : 0);
          const xb = b.box ? b.box[0] : (b.polygon ? Math.min(...b.polygon.map(p => p[0])) : 0);
          return xa - xb;
        }));
        currentRow = [s];
      }
      lastY = y;
    });
    if (currentRow.length > 0) {
      rows.push(currentRow.sort((a, b) => {
        const xa = a.box ? a.box[0] : (a.polygon ? Math.min(...a.polygon.map(p => p[0])) : 0);
        const xb = b.box ? b.box[0] : (b.polygon ? Math.min(...b.polygon.map(p => p[0])) : 0);
        return xa - xb;
      }));
    }
    return rows;
  };

  const dynamicRows = generateDynamicLayout();

  return (
    <div className="app-shell min-h-screen">
      <header className="app-header glass sticky top-0 z-50">
        <div className="flex justify-between items-center w-full px-4 max-w-7xl mx-auto">
          <div>
            <h1>Library Seat Finder</h1>
            <p className="subtitle text-sm opacity-70">Real-time availability monitor</p>
          </div>
          <Link href="/admin" className="text-xs opacity-30 hover:opacity-100 transition-opacity">
            Admin Login
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4 md:p-8">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <div className="spinner" />
            <p>Syncing live status...</p>
          </div>
        ) : totalSeats === 0 ? (
          <div className="config-panel glass text-center p-12">
            <h2 className="text-xl font-medium mb-4">No seats configured</h2>
            <p className="opacity-60 mb-8">The admin hasn't set up any monitoring zones yet.</p>
            <Link href="/admin" className="btn btn-primary">Go to Admin Setup</Link>
          </div>
        ) : (
          <div className="fade-in space-y-12">
            {/* Summary Chips - Pill Style */}
            <div className="flex flex-wrap gap-4 justify-center">
              <div className="summary-chip free-chip px-6 py-2 !bg-transparent flex items-center gap-3">
                <span className="text-2xl font-black">{freeCount}</span>
                <span className="text-[10px] uppercase font-bold tracking-[0.2em] opacity-60">Available</span>
              </div>
              <div className="summary-chip occ-chip px-6 py-2 !bg-transparent flex items-center gap-3">
                <span className="text-2xl font-black">{occupiedCount}</span>
                <span className="text-[10px] uppercase font-bold tracking-[0.2em] opacity-60">Occupied</span>
              </div>
            </div>

            {/* Visual Floor Plan */}
            <div className="floor-plan-container glass">
              <div className="floor-plan-layout">
                {source === 'file' ? (
                  <>
                    {/* BACK ROW */}
                    <div className="seat-row">
                      {['S1', 'S2', 'S3'].map(id => (
                        <SeatPod key={id} id={id} isOccupied={!!status[id]} />
                      ))}
                    </div>

                    {/* MIDDLE AREA: ROUND TABLE + SIDE SEATS */}
                    <div className="round-table-area">
                      <div className="round-table-ui anim-float">
                        <span className="table-label">Central Table</span>
                      </div>

                      {/* SIDE SEATS (ABS POSITIONED ON RIGHT) */}
                      <div className="right-seats">
                        <div className="seat-column">
                          {['S7', 'S8', 'S9'].map(id => (
                            <SeatPod key={id} id={id} isOccupied={!!status[id]} />
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* FRONT ROW */}
                    <div className="seat-row">
                      {['S4', 'S5', 'S6'].map(id => (
                        <SeatPod key={id} id={id} isOccupied={!!status[id]} />
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="dynamic-grid space-y-8">
                    {dynamicRows.map((row, i) => (
                      <div key={i} className="seat-row flex justify-center gap-6 flex-wrap">
                        {row.map(s => (
                          <SeatPod key={s.id} id={s.id} isOccupied={!!status[s.id]} />
                        ))}
                      </div>
                    ))}
                    {dynamicRows.length === 0 && <p className="text-center opacity-40">Arranging seats...</p>}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="py-12 text-center opacity-30 text-[10px] uppercase tracking-widest">
        &copy; {new Date().getFullYear()} AI Seat Detector &bull; Intelligent Library Systems
      </footer>
    </div>
  );
}

function SeatPod({ id, isOccupied }: { id: string; isOccupied: boolean }) {
  return (
    <div className={`seat-pod ${isOccupied ? 'occupied' : 'available'}`}>
      <span className="seat-pod-icon">
        {isOccupied ? '👤' : '▭'}
      </span>
      <span className="seat-pod-id">{id}</span>
      <span className="seat-status-tag">
        {isOccupied ? 'Taken' : 'Free'}
      </span>
    </div>
  );
}
