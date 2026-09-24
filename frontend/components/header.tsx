'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { fetchHealth } from '@/lib/api';
import { HealthResponse } from '@/lib/types';
import { Swords, BarChart3, Zap, BookOpen, CircleDot } from 'lucide-react';

export function Header() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ status: 'offline', models_loaded: 0 }));
    const interval = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => setHealth({ status: 'offline', models_loaded: 0 }));
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const isOnline = health?.status === 'healthy' && health.models_loaded === 5;
  const isPartial = health?.models_loaded && health.models_loaded > 0;

  return (
    <header className="sticky top-0 z-50 border-b border-border-subtle bg-paper-surface/90 backdrop-blur-md">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        {/* Brand Logo */}
        <Link href="/" className="flex items-center gap-3 group hover:no-underline">
          <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 group-hover:border-cyan-400/50 group-hover:scale-105 transition-all">
            <Swords className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-black text-lg tracking-tight text-slate-100 group-hover:text-cyan-300 transition-colors">
                ALGORITHM ARENA
              </span>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-cyan-400 border border-cyan-500/20">
                v1.0
              </span>
            </div>
            <p className="text-[11px] font-mono text-slate-400">
              Shopper Intent Benchmarking · 5 ML Models
            </p>
          </div>
        </Link>

        {/* View Switcher Nav Pills */}
        <div className="flex items-center gap-1 p-1 rounded-full bg-paper-input border border-border-subtle text-xs sm:text-sm">
          <Link
            href="/"
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full font-medium transition-all ${
              pathname === '/'
                ? 'bg-paper-hover text-slate-100 border border-border-subtle shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:no-underline'
            }`}
          >
            <Zap className={`w-4 h-4 ${pathname === '/' ? 'text-cyan-400' : 'text-slate-400'}`} />
            <span>Prediction Arena</span>
          </Link>

          <Link
            href="/report"
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full font-medium transition-all ${
              pathname === '/report'
                ? 'bg-paper-hover text-slate-100 border border-border-subtle shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:no-underline'
            }`}
          >
            <BarChart3 className={`w-4 h-4 ${pathname === '/report' ? 'text-purple-400' : 'text-slate-400'}`} />
            <span>Model Report Card</span>
          </Link>

          <Link
            href="/about"
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full font-medium transition-all ${
              pathname === '/about'
                ? 'bg-paper-hover text-slate-100 border border-border-subtle shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200 hover:no-underline'
            }`}
          >
            <BookOpen className={`w-4 h-4 ${pathname === '/about' ? 'text-amber-400' : 'text-slate-400'}`} />
            <span>About & Algorithms</span>
          </Link>
        </div>

        {/* Health Status Indicator */}
        <div className="flex items-center gap-2 font-mono text-xs text-slate-400 bg-paper-input/50 px-3 py-1.5 rounded-full border border-border-subtle">
          <span
            className={`w-2 h-2 rounded-full transition-all ${
              isOnline
                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-pulse'
                : isPartial
                ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)]'
                : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]'
            }`}
          />
          <span>
            {isOnline
              ? '5/5 Models Active'
              : isPartial
              ? `${health?.models_loaded}/5 Active`
              : 'Backend Offline'}
          </span>
        </div>
      </nav>
    </header>
  );
}

