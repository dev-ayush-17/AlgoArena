'use client';

import { ModelMetricDetail } from '@/lib/types';

interface PerformanceChartsProps {
  models: Array<ModelMetricDetail & { key: string }>;
}

export function PerformanceCharts({ models }: PerformanceChartsProps) {
  const getModelColor = (key: string) => {
    const map: Record<string, string> = {
      logistic_regression: '#38bdf8',
      knn: '#c084fc',
      svm: '#fbbf24',
      decision_tree: '#34d399',
      naive_bayes: '#fb7185',
    };
    return map[key] || '#38bdf8';
  };

  const maxMs = Math.max(...models.map((m) => m.avg_inference_time_ms));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
      {/* Accuracy Comparison Chart */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-6 shadow-lg">
        <h3 className="text-sm font-display font-extrabold text-slate-100 mb-4 flex items-center gap-2">
          <span>🎯 Accuracy Comparison (%)</span>
        </h3>
        <div className="space-y-3">
          {models.map((m) => {
            const accPct = (m.accuracy * 100).toFixed(1);
            return (
              <div key={m.key} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-slate-300">{m.display_name}</span>
                  <span className="font-mono font-bold text-slate-100">{accPct}%</span>
                </div>
                <div className="w-full h-2.5 bg-paper-input rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${accPct}%`,
                      backgroundColor: getModelColor(m.key),
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Latency Speed Benchmark Chart */}
      <div className="bg-paper-surface border border-border-subtle rounded-xl p-6 shadow-lg">
        <h3 className="text-sm font-display font-extrabold text-slate-100 mb-4 flex items-center gap-2">
          <span>⚡ Inference Speed Benchmark (ms)</span>
        </h3>
        <div className="space-y-3">
          {models.map((m) => {
            const latMs = m.avg_inference_time_ms;
            const widthPct = Math.max(5, (latMs / maxMs) * 100).toFixed(1);
            return (
              <div key={m.key} className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-slate-300">{m.display_name}</span>
                  <span className="font-mono font-bold text-slate-100">{latMs.toFixed(3)} ms</span>
                </div>
                <div className="w-full h-2.5 bg-paper-input rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${widthPct}%`,
                      backgroundColor: getModelColor(m.key),
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
