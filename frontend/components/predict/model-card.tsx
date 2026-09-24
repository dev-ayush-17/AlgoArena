'use client';

import { SingleModelPrediction } from '@/lib/types';
import { Zap, Target, Gauge } from 'lucide-react';

interface ModelCardProps {
  prediction: SingleModelPrediction;
  isFastest: boolean;
  isTopConfidence: boolean;
}

const MODEL_STYLE_MAP: Record<string, { accentClass: string; colorHex: string; tag: string }> = {
  logistic_regression: {
    accentClass: 'border-l-cyan-400',
    colorHex: '#38bdf8',
    tag: 'Linear Model',
  },
  knn: {
    accentClass: 'border-l-purple-400',
    colorHex: '#c084fc',
    tag: 'Instance Learner',
  },
  svm: {
    accentClass: 'border-l-amber-400',
    colorHex: '#fbbf24',
    tag: 'Margin Classifier',
  },
  decision_tree: {
    accentClass: 'border-l-emerald-400',
    colorHex: '#34d399',
    tag: 'Non-Linear Tree',
  },
  naive_bayes: {
    accentClass: 'border-l-rose-400',
    colorHex: '#fb7185',
    tag: 'Probabilistic',
  },
};

export function ModelCard({ prediction, isFastest, isTopConfidence }: ModelCardProps) {
  const isPositive = prediction.prediction === 1;
  const confPct = Math.round((prediction.confidence || 0) * 100);
  const styleInfo = MODEL_STYLE_MAP[prediction.model_key] || {
    accentClass: 'border-l-cyan-400',
    colorHex: '#38bdf8',
    tag: 'Classifier',
  };

  return (
    <div
      className={`bg-paper-surface border border-border-subtle rounded-xl p-5 border-l-4 ${styleInfo.accentClass} relative overflow-hidden transition-all hover:border-border-strong hover:shadow-xl`}
    >
      {/* Header Row */}
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="text-base font-display font-bold text-slate-100">
            {prediction.model_name}
          </h3>
          <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
            {styleInfo.tag}
          </span>
        </div>

        <div className="flex flex-col gap-1 items-end">
          {isFastest && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold flex items-center gap-1">
              <Zap className="w-3 h-3" />
              <span>Fastest</span>
            </span>
          )}
          {isTopConfidence && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30 font-semibold flex items-center gap-1">
              <Target className="w-3 h-3" />
              <span>Top Conf.</span>
            </span>
          )}
        </div>
      </div>

      {/* Outcome Pill */}
      <div
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold mb-4 ${
          isPositive
            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
        }`}
      >
        <span>{isPositive ? '🛍️' : '🛑'}</span>
        <span>{prediction.prediction_label}</span>
      </div>

      {/* Confidence Bar Meter */}
      <div className="space-y-1.5 mb-4">
        <div className="flex justify-between items-center text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <Gauge className="w-3.5 h-3.5 text-slate-500" />
            <span>Confidence</span>
          </span>
          <span className="font-mono font-bold text-slate-200">{confPct}%</span>
        </div>
        <div className="w-full h-1.5 bg-paper-input rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${confPct}%`,
              backgroundColor: styleInfo.colorHex,
            }}
          />
        </div>
      </div>

      {/* Latency Footer */}
      <div className="pt-3 border-t border-border-subtle flex justify-between items-center text-xs text-slate-400">
        <span>Inference Latency</span>
        <span className="font-mono font-bold text-slate-100">
          {(prediction.inference_time_ms || 0).toFixed(3)} ms
        </span>
      </div>
    </div>
  );
}
