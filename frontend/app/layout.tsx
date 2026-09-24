import type { Metadata } from 'next';
import './globals.css';
import { Header } from '@/components/header';
import { Cpu, Github, ExternalLink, Activity } from 'lucide-react';

export const metadata: Metadata = {
  title: 'Algorithm Arena — 5-Model Machine Learning Benchmarking',
  description:
    'Interactive machine learning model benchmarking suite comparing Logistic Regression, KNN, SVM, Decision Tree, and Naive Bayes on E-Commerce Shopper Intent Data.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body className="bg-paper text-slate-100 min-h-screen flex flex-col font-body antialiased selection:bg-cyan-500/20 selection:text-cyan-200 overflow-x-clip">
        {/* Top subtle glow banner */}
        <div className="fixed top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-48 bg-gradient-to-b from-cyan-500/5 via-purple-500/5 to-transparent blur-3xl pointer-events-none z-0" />

        <div className="relative z-10 flex flex-col min-h-screen">
          <Header />
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-border-subtle bg-paper-surface/60 mt-16 py-8 px-4 sm:px-6">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6 text-xs text-slate-400">
              <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-lg bg-slate-800 text-cyan-400 border border-border-subtle">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <div className="font-display font-bold text-slate-200 text-sm">
                    Algorithm Arena &middot; ML Benchmarking
                  </div>
                  <p className="text-slate-400">
                    UCI Machine Learning Repository ID 468 &middot; Online Shoppers Purchasing Intention
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-6 font-mono text-[11px]">
                <a
                  href="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 text-slate-400 hover:text-cyan-400 transition-colors"
                >
                  <span>UCI Dataset #468</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
                <span className="text-slate-700">&bull;</span>
                <span className="text-slate-400">5 Models &middot; Scikit-Learn + FastAPI + Next.js</span>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
