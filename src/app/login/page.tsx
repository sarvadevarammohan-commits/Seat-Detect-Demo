'use client';

import { useState } from 'react';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '@/lib/firebase';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.push('/admin');
    } catch (err: any) {
      setError(err.message || 'Failed to login');
      setLoading(false);
    }
  };

  return (
    <div className="app-shell flex items-center justify-center min-h-screen p-4">
      <div className="config-panel glass fade-in max-w-sm w-full" style={{ margin: 'auto' }}>
        <header className="text-center mb-8">
          <h1 className="text-2xl font-bold mb-2">Admin Login</h1>
          <p className="subtitle">Enter your credentials to access the detection dashboard.</p>
        </header>

        <form onSubmit={handleLogin} className="space-y-6">
          <div className="form-group">
            <label htmlFor="email text-sm font-medium">Email Address</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="admin@example.com"
              required
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-cyan-500 transition-all outline-none"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password text-sm font-medium">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-cyan-500 transition-all outline-none"
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary w-full py-3 flex items-center justify-center gap-2"
          >
            {loading ? <div className="spinner-sm" /> : 'Sign In'}
          </button>
        </form>

        <div className="mt-8 text-center">
          <button 
            onClick={() => router.push('/')}
            className="text-sm text-white/50 hover:text-white transition-colors"
          >
            ← Back to Public View
          </button>
        </div>
      </div>

      <style jsx>{`
        .spinner-sm {
          width: 18px;
          height: 18px;
          border: 2px solid rgba(255,255,255,0.1);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
