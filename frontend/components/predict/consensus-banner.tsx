'use client';

import { ConsensusSummary } from '@/lib/types';
import { CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';

interface ConsensusBannerProps {
  consensus: ConsensusSummary | null;
}

export function ConsensusBanner({ consensus }: ConsensusBannerProps) {
  if (!consensus) {
    return (
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 mb-6 flex items-center justify-between opacity-60">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-slate-800 text-slate-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <span className="text-sm text-slate-400">
            Select a preset or click <strong>Benchmark Predictions</strong> to compare all 5 models.
          </span>
        </div>
      </div>
    );
  }

  const isPositive = consensus.majority_prediction === 1;
  const agreementPct = Math.round((consensus.agreement_ratio || 0) * 100);
  const isUnanimous = agreementPct === 100;

  return (
    <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 mb-6 flex flex-wrap items-center justify-between gap-4 shadow-lg">
      <div className="flex items-center gap-4">
        {/* Outcome Badge */}
        <div
          className={`px-4 py-2 rounded-full font-display font-black text-sm tracking-wider uppercase border flex items-center gap-2 ${
            isPositive
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}
        >
          {isPositive ? '🛍️ Purchase Completed' : '🛑 No Purchase'}
        </div>

        {/* Agreement Text */}
        <div className="text-sm text-slate-300">
          <span className="font-bold text-slate-100">
            {consensus.models_agreeing} of {consensus.models_total} models agree
          </span>{' '}
          ({agreementPct}% consensus)
        </div>
      </div>

      {/* Unanimous / Split Tag */}
      <div className="font-mono text-xs flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900 border border-border-subtle text-slate-400">
        {isUnanimous ? (
          <>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-emerald-400 font-semibold">Unanimous Consensus</span>
          </>
        ) : (
          <>
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-amber-400 font-semibold">Split Model Vote</span>
          </>
        )}
      </div>
    </div>
  );
}
