'use client';

import { useState, useEffect } from 'react';
import { FormSchemaResponse } from '@/lib/types';
import { Sliders, Sparkles, Flame, LogOut, Search } from 'lucide-react';

interface DynamicFormProps {
  schema: FormSchemaResponse;
  onSubmit: (features: Record<string, any>) => void;
  isLoading: boolean;
}

const PRESETS = {
  high_intent: {
    name: 'High-Intent Shopper',
    icon: Flame,
    features: {
      Administrative: 4,
      Administrative_Duration: 120.0,
      Informational: 2,
      Informational_Duration: 60.0,
      ProductRelated: 35,
      ProductRelated_Duration: 1450.0,
      BounceRates: 0.0,
      ExitRates: 0.01,
      PageValues: 45.2,
      SpecialDay: 0.0,
      Month: 'Nov',
      OperatingSystems: 2,
      Browser: 2,
      Region: 1,
      TrafficType: 2,
      VisitorType: 'Returning_Visitor',
      Weekend: 1,
    },
  },
  bouncer: {
    name: 'Immediate Bouncer',
    icon: LogOut,
    features: {
      Administrative: 0,
      Administrative_Duration: 0.0,
      Informational: 0,
      Informational_Duration: 0.0,
      ProductRelated: 1,
      ProductRelated_Duration: 0.0,
      BounceRates: 0.2,
      ExitRates: 0.2,
      PageValues: 0.0,
      SpecialDay: 0.0,
      Month: 'May',
      OperatingSystems: 1,
      Browser: 1,
      Region: 3,
      TrafficType: 1,
      VisitorType: 'New_Visitor',
      Weekend: 0,
    },
  },
  researcher: {
    name: 'Deep Researcher',
    icon: Search,
    features: {
      Administrative: 6,
      Administrative_Duration: 240.0,
      Informational: 4,
      Informational_Duration: 180.0,
      ProductRelated: 60,
      ProductRelated_Duration: 2800.0,
      BounceRates: 0.01,
      ExitRates: 0.03,
      PageValues: 5.0,
      SpecialDay: 0.2,
      Month: 'Dec',
      OperatingSystems: 3,
      Browser: 2,
      Region: 2,
      TrafficType: 3,
      VisitorType: 'Returning_Visitor',
      Weekend: 0,
    },
  },
};

export function DynamicForm({ schema, onSubmit, isLoading }: DynamicFormProps) {
  const [formValues, setFormValues] = useState<Record<string, any>>({});

  // Populate initial defaults
  useEffect(() => {
    const defaults: Record<string, any> = {};
    schema.numerical_features?.forEach((f) => {
      defaults[f.name] = f.default !== undefined ? f.default : 0;
    });
    schema.categorical_features?.forEach((f) => {
      defaults[f.name] =
        f.default !== undefined
          ? f.default
          : f.categories
          ? f.categories[0]
          : '';
    });
    setFormValues(defaults);
  }, [schema]);

  const handleChange = (name: string, value: any) => {
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const applyPreset = (presetKey: keyof typeof PRESETS) => {
    setFormValues(PRESETS[presetKey].features);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Ensure proper types
    const cleanedPayload: Record<string, any> = {};
    schema.numerical_features?.forEach((f) => {
      const val = formValues[f.name];
      cleanedPayload[f.name] =
        f.dtype === 'int64' ? parseInt(val, 10) : parseFloat(val);
    });
    schema.categorical_features?.forEach((f) => {
      const val = formValues[f.name];
      cleanedPayload[f.name] = f.dtype === 'int64' ? parseInt(val, 10) : val;
    });

    onSubmit(cleanedPayload);
  };

  return (
    <div className="bg-paper-surface border border-border-subtle rounded-xl p-6 shadow-xl">
      <div className="border-b border-border-subtle pb-4 mb-6">
        <h2 className="text-xl font-display font-extrabold flex items-center gap-2">
          <Sliders className="w-5 h-5 text-cyan-400" />
          <span>Session Feature Inputs</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Configure session behavior metrics or load a pre-configured preset.
        </p>
      </div>

      {/* Preset Toolbar */}
      <div className="mb-6">
        <span className="text-xs font-mono uppercase tracking-wider text-slate-500 block mb-2">
          Quick Presets
        </span>
        <div className="flex flex-wrap gap-2">
          {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((key) => {
            const P = PRESETS[key];
            const Icon = P.icon;
            return (
              <button
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-paper-card border border-border-subtle text-xs font-semibold text-slate-300 hover:bg-paper-hover hover:border-slate-500 hover:text-slate-100 transition-all"
              >
                <Icon className="w-3.5 h-3.5 text-cyan-400" />
                <span>{P.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Numerical Group */}
        {schema.numerical_features?.length > 0 && (
          <div>
            <div className="font-mono text-[11px] uppercase tracking-widest text-slate-500 mb-3 font-semibold">
              Browsing Behavior Metrics
            </div>
            <div className="space-y-3">
              {schema.numerical_features.map((feat) => {
                const val = formValues[feat.name] ?? feat.default ?? 0;
                const isRate = feat.max && feat.max <= 1.0;

                return (
                  <div key={feat.name} className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <label htmlFor={feat.name} className="font-medium text-slate-200">
                        {feat.name.replace(/_/g, ' ')}
                      </label>
                      <span className="font-mono text-[11px] text-cyan-400 bg-slate-900 px-2 py-0.5 rounded border border-border-subtle">
                        {val}
                      </span>
                    </div>
                    <input
                      type={isRate ? 'range' : 'number'}
                      id={feat.name}
                      name={feat.name}
                      value={val}
                      min={feat.min ?? 0}
                      max={feat.max ?? 100}
                      step={feat.dtype === 'float64' ? 0.01 : 1}
                      onChange={(e) => handleChange(feat.name, e.target.value)}
                      className="w-full bg-paper-input border border-border-subtle rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 transition-colors"
                      required
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Categorical Group */}
        {schema.categorical_features?.length > 0 && (
          <div className="pt-2">
            <div className="font-mono text-[11px] uppercase tracking-widest text-slate-500 mb-3 font-semibold">
              Session Metadata
            </div>
            <div className="space-y-3">
              {schema.categorical_features.map((feat) => {
                const val = formValues[feat.name] ?? feat.default ?? '';

                return (
                  <div key={feat.name} className="space-y-1">
                    <label htmlFor={feat.name} className="block text-xs font-medium text-slate-200">
                      {feat.name}
                    </label>
                    <select
                      id={feat.name}
                      name={feat.name}
                      value={val}
                      onChange={(e) => handleChange(feat.name, e.target.value)}
                      className="w-full bg-paper-input border border-border-subtle rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 transition-colors"
                    >
                      {feat.categories?.map((cat) => (
                        <option key={String(cat)} value={String(cat)}>
                          {String(cat)}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="w-full mt-6 bg-slate-100 hover:bg-white text-slate-950 font-display font-extrabold text-sm py-3.5 px-4 rounded-lg shadow-lg hover:shadow-cyan-500/10 active:translate-y-0.5 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
              <span>Executing 5-Model Inference...</span>
            </span>
          ) : (
            <>
              <Sparkles className="w-4 h-4 text-cyan-600" />
              <span>Benchmark Predictions across 5 Models</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
