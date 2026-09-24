'use client';

import { useState } from 'react';
import { SingleModelPrediction } from '@/lib/types';
import { ModelCard } from './model-card';
import { ArrowUpDown } from 'lucide-react';

interface ArenaGridProps {
  predictions: SingleModelPrediction[];
}

export function ArenaGrid({ predictions }: ArenaGridProps) {
  const [sortBy, setSortBy] = useState<'default' | 'latency' | 'confidence'>('default');

  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-center py-16 px-6 bg-paper-surface border border-dashed border-border-subtle rounded-xl">
        <p className="text-slate-400 text-sm">
          Submit session features on the left to benchmark 5 models side-by-side.
        </p>
      </div>
    );
  }

  // Calculate fastest & highest confidence
  const latencies = predictions.map((p) => p.inference_time_ms).filter((l) => typeof l === 'number');
  const minLatency = Math.min(...latencies);

  const confidences = predictions.map((p) => p.confidence).filter((c) => typeof c === 'number');
  const maxConfidence = Math.max(...confidences);

  // Sorting logic
  const sorted = [...predictions].sort((a, b) => {
    if (sortBy === 'latency') return a.inference_time_ms - b.inference_time_ms;
    if (sortBy === 'confidence') return b.confidence - a.confidence;
    return 0;
  });

  return (
    <div>
      {/* Sorting Header */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-display font-extrabold text-slate-100">
          Side-by-Side Model Arena
        </h2>
        <div className="flex items-center gap-2">
          <label htmlFor="sort-select" className="text-xs text-slate-400 flex items-center gap-1">
            <ArrowUpDown className="w-3.5 h-3.5" />
            <span>Sort:</span>
          </label>
          <select
            id="sort-select"
            value={sortBy}
            onChange={(e: any) => setSortBy(e.target.value)}
            className="bg-paper-input border border-border-subtle rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="default">Default Order</option>
            <option value="latency">Fastest Speed</option>
            <option value="confidence">Top Confidence</option>
          </select>
        </div>
      </div>

      {/* Grid Container */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {sorted.map((pred) => (
          <ModelCard
            key={pred.model_key}
            prediction={pred}
            isFastest={pred.inference_time_ms === minLatency}
            isTopConfidence={pred.confidence === maxConfidence}
          />
        ))}
      </div>
    </div>
  );
}
