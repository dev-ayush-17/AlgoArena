'use client';

import { useState, useEffect } from 'react';
import { fetchFormSchema, submitPrediction } from '@/lib/api';
import { FormSchemaResponse, PredictionResponse } from '@/lib/types';
import { DynamicForm } from '@/components/predict/dynamic-form';
import { ConsensusBanner } from '@/components/predict/consensus-banner';
import { ArenaGrid } from '@/components/predict/arena-grid';
import { Zap, Activity, AlertCircle, Info, Sparkles, Layers } from 'lucide-react';

export default function PredictionPage() {
  const [schema, setSchema] = useState<FormSchemaResponse | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoadingSchema, setIsLoadingSchema] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadSchema() {
      try {
        setIsLoadingSchema(true);
        setError(null);
        const schemaData = await fetchFormSchema();
        setSchema(schemaData);
      } catch (err: any) {
        console.error('Failed to load schema:', err);
        setError('Failed to load feature schema from backend server. Using fallback features.');
      } finally {
        setIsLoadingSchema(false);
      }
    }
    loadSchema();
  }, []);

  const handlePredict = async (features: Record<string, any>) => {
    try {
      setIsPredicting(true);
      setError(null);
      const res = await submitPrediction(features);
      setPredictionResult(res);
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.message || 'An error occurred while executing inference models.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Hero Header */}
      <div className="bg-paper-surface/80 border border-border-subtle rounded-2xl p-6 sm:p-8 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
        
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="max-w-2xl space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 font-mono text-xs text-cyan-400 font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Interactive Model Benchmark Arena</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-display font-black text-slate-100 tracking-tight">
              Compare 5 Machine Learning Models Side-by-Side
            </h1>
            <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
              Submit real e-commerce browsing metrics to evaluate predictions, confidence scores, and inference latency across <strong>Logistic Regression</strong>, <strong>KNN</strong>, <strong>SVM</strong>, <strong>Decision Tree</strong>, and <strong>Naive Bayes</strong> in real time.
            </p>
          </div>

          {/* Quick Metrics Badge Strip */}
          <div className="grid grid-cols-2 gap-3 sm:gap-4 shrink-0 font-mono">
            <div className="bg-paper-card p-3.5 rounded-xl border border-border-subtle">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400 uppercase">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>Algorithms</span>
              </div>
              <div className="text-xl font-bold text-slate-100 mt-1">5 Models</div>
            </div>
            <div className="bg-paper-card p-3.5 rounded-xl border border-border-subtle">
              <div className="flex items-center gap-1.5 text-[11px] text-slate-400 uppercase">
                <Zap className="w-3.5 h-3.5 text-emerald-400" />
                <span>Latency</span>
              </div>
              <div className="text-xl font-bold text-emerald-400 mt-1">&lt; 1 ms</div>
            </div>
          </div>
        </div>
      </div>

      {/* Error Notification Alert */}
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="font-semibold text-rose-200">Execution Error</div>
            <p className="text-xs text-rose-300/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Main Grid: Left Form / Right Arena */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Feature Form */}
        <div className="lg:col-span-5">
          {isLoadingSchema ? (
            <div className="bg-paper-surface border border-border-subtle rounded-xl p-8 text-center space-y-4">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs font-mono text-slate-400">Loading feature schema definition...</p>
            </div>
          ) : schema ? (
            <DynamicForm
              schema={schema}
              onSubmit={handlePredict}
              isLoading={isPredicting}
            />
          ) : (
            <div className="bg-paper-surface border border-border-subtle rounded-xl p-6 text-center text-slate-400 text-xs">
              Schema missing. Please ensure FastAPI backend is running at http://127.0.0.1:8000.
            </div>
          )}
        </div>

        {/* Right Column: Consensus + Model Arena Cards */}
        <div className="lg:col-span-7 space-y-6">
          <ConsensusBanner consensus={predictionResult?.consensus || null} />
          <ArenaGrid predictions={predictionResult?.predictions || []} />
        </div>
      </div>
    </div>
  );
}
