'use client';

import { useState, useEffect } from 'react';
import { fetchMetrics } from '@/lib/api';
import { MetricsResponse, ModelMetricDetail } from '@/lib/types';
import { LeaderboardSummary } from '@/components/report/leaderboard-summary';
import { MetricsTable } from '@/components/report/metrics-table';
import { PerformanceCharts } from '@/components/report/performance-charts';
import { NarrativeCards } from '@/components/report/narrative-cards';
import { BarChart3, Database, Trophy, Sparkles, RefreshCw } from 'lucide-react';

export default function ReportPage() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const metrics = await fetchMetrics();
      setData(metrics);
    } catch (err: any) {
      console.error('Failed to load metrics:', err);
      setError('Could not load offline evaluation benchmark metrics from backend server.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (isLoading) {
    return (
      <div className="py-20 text-center space-y-4">
        <div className="w-10 h-10 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-xs font-mono text-slate-400">Loading benchmark evaluation metrics...</p>
      </div>
    );
  }

  if (error || !data || !data.models) {
    return (
      <div className="py-16 text-center space-y-4 bg-paper-surface border border-border-subtle rounded-2xl p-8 max-w-xl mx-auto">
        <div className="text-amber-400 font-bold text-lg">Unable to Load Benchmark Metrics</div>
        <p className="text-xs text-slate-400">
          {error || 'Metrics data is currently unavailable. Please verify backend execution state.'}
        </p>
        <button
          onClick={loadData}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-paper-hover border border-border-subtle text-xs font-semibold text-slate-200 hover:text-white"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      </div>
    );
  }

  // Convert models map into array
  const modelList: Array<ModelMetricDetail & { key: string }> = Object.entries(data.models).map(
    ([key, item]) => ({
      key,
      ...item,
    })
  );

  // Calculate bests & worsts
  const findBest = (metricKey: keyof ModelMetricDetail, lowestIsBest = false) => {
    return [...modelList].sort((a, b) => {
      const valA = (a[metricKey] as number) || 0;
      const valB = (b[metricKey] as number) || 0;
      return lowestIsBest ? valA - valB : valB - valA;
    })[0];
  };

  const findWorst = (metricKey: keyof ModelMetricDetail, lowestIsWorst = false) => {
    return [...modelList].sort((a, b) => {
      const valA = (a[metricKey] as number) || 0;
      const valB = (b[metricKey] as number) || 0;
      return lowestIsWorst ? valA - valB : valB - valA;
    })[0];
  };

  const bestsMap = {
    accuracy: findBest('accuracy').key,
    precision: findBest('precision').key,
    recall: findBest('recall').key,
    f1_score: findBest('f1_score').key,
    avg_inference_time_ms: findBest('avg_inference_time_ms', true).key,
    model_file_size_kb: findBest('model_file_size_kb', true).key,
  };

  const worstsMap = {
    accuracy: findWorst('accuracy', true).key,
    precision: findWorst('precision', true).key,
    recall: findWorst('recall', true).key,
    f1_score: findWorst('f1_score', true).key,
    avg_inference_time_ms: findWorst('avg_inference_time_ms').key,
    model_file_size_kb: findWorst('model_file_size_kb').key,
  };

  const bestsObjects = {
    accuracy: findBest('accuracy'),
    f1_score: findBest('f1_score'),
    avg_inference_time_ms: findBest('avg_inference_time_ms', true),
    model_file_size_kb: findBest('model_file_size_kb', true),
  };

  const worstsObjects = {
    accuracy: findWorst('accuracy', true),
    avg_inference_time_ms: findWorst('avg_inference_time_ms'),
    model_file_size_kb: findWorst('model_file_size_kb'),
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="bg-paper-surface/80 border border-border-subtle rounded-2xl p-6 sm:p-8 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="max-w-2xl space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 font-mono text-xs text-purple-400 font-semibold">
              <Trophy className="w-3.5 h-3.5" />
              <span>Model Evaluation & Trade-off Report</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-display font-black text-slate-100 tracking-tight">
              Algorithm Performance Report Card
            </h1>
            <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
              Comprehensive test set benchmarks evaluating <strong>Accuracy</strong>, <strong>Precision</strong>, <strong>Recall</strong>, <strong>F1 Score</strong>, <strong>Inference Latency</strong>, and <strong>Disk Footprint</strong> across all 5 trained models.
            </p>
          </div>

          {/* Dataset Info Box */}
          {data.dataset_size && (
            <div className="bg-paper-card border border-border-subtle p-4 rounded-xl text-xs space-y-1.5 shrink-0 font-mono">
              <div className="flex items-center gap-2 text-slate-400 font-semibold text-[11px] uppercase">
                <Database className="w-3.5 h-3.5 text-purple-400" />
                <span>Dataset Summary</span>
              </div>
              <div className="text-slate-200">
                Total Rows: <strong className="text-slate-100">{data.dataset_size.total_rows?.toLocaleString() || 12330}</strong>
              </div>
              <div className="text-slate-400">
                Split: {data.dataset_size.train_rows?.toLocaleString() || 9864} train / {data.dataset_size.test_rows?.toLocaleString() || 2466} test
              </div>
              <div className="text-slate-400">
                Features: {data.dataset_size.features || 17} raw &bull; {data.dataset_size.features_after_encoding || 68} encoded
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Top Cards: Bests Leaderboard */}
      <LeaderboardSummary bests={bestsObjects} />

      {/* Trade-Off Narrative Insight Cards */}
      <div>
        <h2 className="text-lg font-display font-black text-slate-100 mb-4 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span>Automated Trade-Off Insights</span>
        </h2>
        <NarrativeCards bests={bestsObjects} worsts={worstsObjects} />
      </div>

      {/* Performance Bar Charts */}
      <div>
        <h2 className="text-lg font-display font-black text-slate-100 mb-4 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <span>Comparative Metrics Visualization</span>
        </h2>
        <PerformanceCharts models={modelList} />
      </div>

      {/* Detailed Metrics Table */}
      <div>
        <h2 className="text-lg font-display font-black text-slate-100 mb-4">
          Complete Metrics Matrix
        </h2>
        <MetricsTable models={modelList} bests={bestsMap} worsts={worstsMap} />
      </div>
    </div>
  );
}
