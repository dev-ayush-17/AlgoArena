'use client';

import { ModelMetricDetail } from '@/lib/types';

interface MetricsTableProps {
  models: Array<ModelMetricDetail & { key: string }>;
  bests: Record<string, string>;
  worsts: Record<string, string>;
}

export function MetricsTable({ models, bests, worsts }: MetricsTableProps) {
  const formatKB = (kb: number) => {
    return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`;
  };

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

  return (
    <div className="overflow-x-auto border border-border-subtle rounded-xl mb-8 shadow-xl">
      <table className="w-full text-left text-sm border-collapse">
        <thead>
          <tr className="bg-paper-surface border-b border-border-subtle font-mono text-[11px] uppercase tracking-wider text-slate-400">
            <th className="py-3.5 px-4 font-semibold">Model Algorithm</th>
            <th className="py-3.5 px-4 font-semibold">Accuracy</th>
            <th className="py-3.5 px-4 font-semibold">Precision</th>
            <th className="py-3.5 px-4 font-semibold">Recall</th>
            <th className="py-3.5 px-4 font-semibold">F1 Score</th>
            <th className="py-3.5 px-4 font-semibold">Avg Latency</th>
            <th className="py-3.5 px-4 font-semibold">Model Size</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle bg-paper-surface/50">
          {models.map((m) => {
            const isBestAcc = m.key === bests.accuracy;
            const isWorstAcc = m.key === worsts.accuracy;

            const isBestPrec = m.key === bests.precision;
            const isWorstPrec = m.key === worsts.precision;

            const isBestRec = m.key === bests.recall;
            const isWorstRec = m.key === worsts.recall;

            const isBestF1 = m.key === bests.f1_score;
            const isWorstF1 = m.key === worsts.f1_score;

            const isBestLat = m.key === bests.avg_inference_time_ms;
            const isWorstLat = m.key === worsts.avg_inference_time_ms;

            const isBestSize = m.key === bests.model_file_size_kb;
            const isWorstSize = m.key === worsts.model_file_size_kb;

            return (
              <tr key={m.key} className="hover:bg-paper-hover transition-colors">
                <td className="py-3.5 px-4 font-bold text-slate-100 flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full inline-block"
                    style={{ backgroundColor: getModelColor(m.key) }}
                  />
                  <span>{m.display_name}</span>
                </td>

                <td
                  className={`py-3.5 px-4 ${
                    isBestAcc
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstAcc
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {m.accuracy.toFixed(3)}
                </td>

                <td
                  className={`py-3.5 px-4 ${
                    isBestPrec
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstPrec
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {m.precision.toFixed(3)}
                </td>

                <td
                  className={`py-3.5 px-4 ${
                    isBestRec
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstRec
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {m.recall.toFixed(3)}
                </td>

                <td
                  className={`py-3.5 px-4 ${
                    isBestF1
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstF1
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {m.f1_score.toFixed(3)}
                </td>

                <td
                  className={`py-3.5 px-4 font-mono ${
                    isBestLat
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstLat
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {m.avg_inference_time_ms.toFixed(3)} ms
                </td>

                <td
                  className={`py-3.5 px-4 font-mono ${
                    isBestSize
                      ? 'font-bold text-emerald-400 bg-emerald-500/10'
                      : isWorstSize
                      ? 'text-slate-500'
                      : 'text-slate-200'
                  }`}
                >
                  {formatKB(m.model_file_size_kb)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
