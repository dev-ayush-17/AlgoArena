'use client';

import { ModelMetricDetail } from '@/lib/types';
import { Zap, Target, Trophy, HardDrive } from 'lucide-react';

interface LeaderboardProps {
  bests: {
    accuracy: ModelMetricDetail & { key: string };
    f1_score: ModelMetricDetail & { key: string };
    avg_inference_time_ms: ModelMetricDetail & { key: string };
    model_file_size_kb: ModelMetricDetail & { key: string };
  };
}

export function LeaderboardSummary({ bests }: LeaderboardProps) {
  const formatKB = (kb: number) => {
    return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`;
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      {/* Fastest */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span>Fastest Inference</span>
        </div>
        <div className="text-2xl font-display font-extrabold text-slate-100 mb-1">
          {bests.avg_inference_time_ms.avg_inference_time_ms.toFixed(3)} ms
        </div>
        <div className="text-xs font-semibold text-emerald-400">
          {bests.avg_inference_time_ms.display_name}
        </div>
      </div>

      {/* Highest Accuracy */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
          <Target className="w-4 h-4 text-cyan-400" />
          <span>Highest Accuracy</span>
        </div>
        <div className="text-2xl font-display font-extrabold text-slate-100 mb-1">
          {(bests.accuracy.accuracy * 100).toFixed(1)}%
        </div>
        <div className="text-xs font-semibold text-cyan-400">
          {bests.accuracy.display_name}
        </div>
      </div>

      {/* Highest F1 */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
          <Trophy className="w-4 h-4 text-amber-400" />
          <span>Highest F1 Score</span>
        </div>
        <div className="text-2xl font-display font-extrabold text-slate-100 mb-1">
          {bests.f1_score.f1_score.toFixed(3)}
        </div>
        <div className="text-xs font-semibold text-amber-400">
          {bests.f1_score.display_name}
        </div>
      </div>

      {/* Smallest Footprint */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
          <HardDrive className="w-4 h-4 text-purple-400" />
          <span>Smallest Footprint</span>
        </div>
        <div className="text-2xl font-display font-extrabold text-slate-100 mb-1">
          {formatKB(bests.model_file_size_kb.model_file_size_kb)}
        </div>
        <div className="text-xs font-semibold text-purple-400">
          {bests.model_file_size_kb.display_name}
        </div>
      </div>
    </div>
  );
}
