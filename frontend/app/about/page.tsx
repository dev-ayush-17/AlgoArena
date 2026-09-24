import Link from 'next/link';
import {
  BookOpen,
  Cpu,
  Layers,
  Zap,
  Target,
  ShieldCheck,
  Code2,
  ExternalLink,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  BarChart3,
} from 'lucide-react';

const ALGORITHMS = [
  {
    key: 'logistic_regression',
    name: 'Logistic Regression',
    category: 'Linear Probabilistic Classifier',
    colorHex: '#38bdf8',
    borderColor: 'border-l-cyan-400',
    tagColor: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    summary:
      'A fundamental linear model that applies the sigmoid function to a weighted linear combination of feature values to model binary outcome probabilities.',
    strengths: [
      'Extremely fast inference latency (< 0.02 ms per row)',
      'Highly interpretable via feature coefficients',
      'Minimal memory & disk footprint (~ 1.7 KB)',
    ],
    weaknesses: [
      'Assumes linear decision boundaries between classes',
      'Can struggle with complex non-linear feature interactions',
    ],
  },
  {
    key: 'knn',
    name: 'K-Nearest Neighbors (KNN)',
    category: 'Non-Parametric Instance Learner',
    colorHex: '#c084fc',
    borderColor: 'border-l-purple-400',
    tagColor: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    summary:
      'A lazy learner that classifies new sessions by computing Euclidean distances to all stored training instances and taking a majority vote of the K nearest neighbors.',
    strengths: [
      'No explicit training phase required',
      'Adapts naturally to complex, non-linear decision surfaces',
    ],
    weaknesses: [
      'Heavy disk & memory footprint (~ 2.3 MB) because it stores all training data',
      'Inference latency scales with dataset size O(N)',
    ],
  },
  {
    key: 'svm',
    name: 'Support Vector Machine (SVM)',
    category: 'Maximum-Margin Convex Classifier',
    colorHex: '#fbbf24',
    borderColor: 'border-l-amber-400',
    tagColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    summary:
      'Finds the optimal hyperplane that maximizes the margin of separation between classes in feature space using support vectors.',
    strengths: [
      'Effective in high-dimensional feature spaces',
      'Robust against overfitting when using regularization (C parameter)',
    ],
    weaknesses: [
      'Longer training times on large datasets',
      'Requires calibrated probabilities via Platt scaling for confidence outputs',
    ],
  },
  {
    key: 'decision_tree',
    name: 'Decision Tree (CART)',
    category: 'Hierarchical Rule-Based Model',
    colorHex: '#34d399',
    borderColor: 'border-l-emerald-400',
    tagColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    summary:
      'Recursively splits feature space based on Gini impurity or entropy criteria to build an interpretable decision tree of IF-THEN rules.',
    strengths: [
      'Requires zero feature scaling prior to training',
      'Captures non-linear relationships and feature interactions intuitively',
    ],
    weaknesses: [
      'Prone to overfitting without depth pruning (`max_depth` limits)',
      'Sensitive to small perturbations in training data',
    ],
  },
  {
    key: 'naive_bayes',
    name: 'Gaussian Naive Bayes',
    category: 'Bayesian Probabilistic Model',
    colorHex: '#fb7185',
    borderColor: 'border-l-rose-400',
    tagColor: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    summary:
      'Applies Bayes Theorem assuming conditional independence between features given the target class label.',
    strengths: [
      'Ultra-fast training and inference speeds',
      'Works surprisingly well even when feature independence assumption is violated',
    ],
    weaknesses: [
      'Can produce skewed confidence probabilities if features are highly correlated',
    ],
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-12 max-w-5xl mx-auto">
      {/* Hero Banner */}
      <div className="bg-paper-surface/80 border border-border-subtle rounded-2xl p-6 sm:p-10 relative overflow-hidden shadow-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />

        <div className="space-y-4 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 font-mono text-xs text-amber-400 font-semibold">
            <BookOpen className="w-3.5 h-3.5" />
            <span>Algorithm Arena Technical Guide</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-display font-black text-slate-100 tracking-tight">
            Understanding Machine Learning Trade-Offs
          </h1>

          <p className="text-base text-slate-300 leading-relaxed max-w-3xl">
            <strong>Algorithm Arena</strong> is an interactive machine learning benchmarking suite designed to demonstrate empirical performance trade-offs between classic classification algorithms on e-commerce shopper intention data.
          </p>
        </div>
      </div>

      {/* Dataset & Problem Context Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-paper-surface border border-border-subtle rounded-2xl p-6 shadow-lg space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-display font-bold text-lg">
            <Layers className="w-5 h-5" />
            <span>The Dataset: UCI ID 468</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            The dataset consists of <strong>12,330 e-commerce sessions</strong> from the UCI Machine Learning Repository. Each session tracks user browsing behavior including administrative, informational, and product-related pageviews, exit rates, bounce rates, page values, special day proximity, operating system, region, and visitor type.
          </p>
          <div className="pt-2 font-mono text-xs text-slate-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span>Target: Revenue (Purchased vs No Purchase)</span>
          </div>
        </div>

        <div className="bg-paper-surface border border-border-subtle rounded-2xl p-6 shadow-lg space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-display font-bold text-lg">
            <ShieldCheck className="w-5 h-5" />
            <span>Pipeline Integrity & Zero Data Leakage</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            All preprocessing transformers (<strong>StandardScaler</strong> for numerical continuous metrics and <strong>OneHotEncoder</strong> for categorical variables) are fitted strictly on the training partition. Serialized artifacts (`scaler.pkl`, `encoder.pkl`) ensure seamless real-time feature transformation during FastAPI inference.
          </p>
          <div className="pt-2 font-mono text-xs text-slate-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>80/20 Train/Test Stratified Split</span>
          </div>
        </div>
      </div>

      {/* Algorithm Deep Dives */}
      <div className="space-y-6">
        <div className="border-b border-border-subtle pb-3">
          <h2 className="text-2xl font-display font-black text-slate-100 flex items-center gap-2">
            <Cpu className="w-6 h-6 text-cyan-400" />
            <span>The 5 Arena Algorithms</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Detailed mathematical intuition, strengths, and limitations for each model evaluated in the arena.
          </p>
        </div>

        <div className="space-y-4">
          {ALGORITHMS.map((algo) => (
            <div
              key={algo.key}
              className={`bg-paper-surface border border-border-subtle rounded-2xl p-6 border-l-4 ${algo.borderColor} shadow-lg space-y-4`}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="space-y-1">
                  <h3 className="text-xl font-display font-bold text-slate-100">
                    {algo.name}
                  </h3>
                  <span className={`inline-block text-[11px] font-mono px-2.5 py-0.5 rounded-full border ${algo.tagColor}`}>
                    {algo.category}
                  </span>
                </div>
              </div>

              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                {algo.summary}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-border-subtle text-xs">
                <div>
                  <span className="font-mono text-[11px] uppercase tracking-wider text-emerald-400 font-semibold block mb-2">
                    Key Advantages
                  </span>
                  <ul className="space-y-1.5 text-slate-300">
                    {algo.strengths.map((st, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                        <span>{st}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <span className="font-mono text-[11px] uppercase tracking-wider text-amber-400 font-semibold block mb-2">
                    Known Trade-offs
                  </span>
                  <ul className="space-y-1.5 text-slate-300">
                    {algo.weaknesses.map((wk, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0 mt-1.5" />
                        <span>{wk}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Action CTA */}
      <div className="bg-gradient-to-r from-cyan-950/40 via-paper-surface to-purple-950/40 border border-border-subtle rounded-2xl p-8 text-center space-y-4 shadow-xl">
        <h3 className="text-2xl font-display font-black text-slate-100">
          Ready to test these models live?
        </h3>
        <p className="text-xs sm:text-sm text-slate-300 max-w-xl mx-auto">
          Jump into the Prediction Arena to tweak browsing metrics and compare live predictions, or inspect the complete offline evaluation report card.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-display font-bold text-xs transition-all shadow-lg shadow-cyan-500/20"
          >
            <Zap className="w-4 h-4" />
            <span>Launch Prediction Arena</span>
          </Link>
          <Link
            href="/report"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-paper-hover border border-border-subtle text-slate-200 hover:text-white font-display font-bold text-xs transition-all"
          >
            <BarChart3 className="w-4 h-4 text-purple-400" />
            <span>View Model Report Card</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
