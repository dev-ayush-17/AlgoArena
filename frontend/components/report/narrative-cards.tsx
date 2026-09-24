'use client';

import { ModelMetricDetail } from '@/lib/types';
import { Zap, HardDrive, Scale } from 'lucide-react';

interface NarrativeCardsProps {
  bests: {
    accuracy: ModelMetricDetail & { key: string };
    f1_score: ModelMetricDetail & { key: string };
    avg_inference_time_ms: ModelMetricDetail & { key: string };
    model_file_size_kb: ModelMetricDetail & { key: string };
  };
  worsts: {
    accuracy: ModelMetricDetail & { key: string };
    avg_inference_time_ms: ModelMetricDetail & { key: string };
    model_file_size_kb: ModelMetricDetail & { key: string };
  };
}

export function NarrativeCards({ bests, worsts }: NarrativeCardsProps) {
  const formatKB = (kb: number) => {
    return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`;
  };

  const speedRatio = Math.round(
    worsts.avg_inference_time_ms.avg_inference_time_ms /
      Math.max(0.001, bests.avg_inference_time_ms.avg_inference_time_ms)
  );

  const sizeRatio = Math.round(
    worsts.model_file_size_kb.model_file_size_kb /
      Math.max(0.1, bests.model_file_size_kb.model_file_size_kb)
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {/* Speed Dynamics Card */}
      <div className="bg-paper-surface border border-border-subtle border-l-4 border-l-emerald-400 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-100 mb-2">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span>Inference Speed Dynamics</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          <strong className="text-slate-100">{bests.avg_inference_time_ms.display_name}</strong>{' '}
          delivers the fastest single-row prediction at{' '}
          <code className="text-emerald-400 font-mono">
            {bests.avg_inference_time_ms.avg_inference_time_ms.toFixed(3)}ms
          </code>
          , executing up to{' '}
          <strong className="text-emerald-400">{speedRatio}x faster</strong> than{' '}
          {worsts.avg_inference_time_ms.display_name} (
          {worsts.avg_inference_time_ms.avg_inference_time_ms.toFixed(3)}ms).
        </p>
      </div>

      {/* Storage Footprint Card */}
      <div className="bg-paper-surface border border-border-subtle border-l-4 border-l-purple-400 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-100 mb-2">
          <HardDrive className="w-4 h-4 text-purple-400" />
          <span>Deployment Footprint</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          <strong className="text-slate-100">{bests.model_file_size_kb.display_name}</strong>{' '}
          requires just{' '}
          <code className="text-purple-400 font-mono">
            {formatKB(bests.model_file_size_kb.model_file_size_kb)}
          </code>{' '}
          disk space, whereas{' '}
          <strong className="text-slate-100">{worsts.model_file_size_kb.display_name}</strong> is{' '}
          <strong className="text-purple-400">{sizeRatio}x larger</strong> (
          {formatKB(worsts.model_file_size_kb.model_file_size_kb)}) because it retains training sample instances.
        </p>
      </div>

      {/* Trade-Off Matrix Card */}
      <div className="bg-paper-surface border border-border-subtle border-l-4 border-l-amber-400 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-100 mb-2">
          <Scale className="w-4 h-4 text-amber-400" />
          <span>Accuracy vs Complexity</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          While <strong className="text-slate-100">{bests.accuracy.display_name}</strong> achieves
          peak accuracy of{' '}
          <code className="text-amber-400 font-mono">
            {(bests.accuracy.accuracy * 100).toFixed(1)}%
          </code>
          , linear and tree models offer near-instant prediction with minimal storage overhead. Model choice depends on real-time SLAs.
        </p>
      </div>
    </div>
  );
}
