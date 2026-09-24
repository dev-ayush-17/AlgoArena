import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: 'oklch(0.14 0.02 250)',
          surface: 'oklch(0.17 0.025 250)',
          card: 'oklch(0.19 0.03 250)',
          hover: 'oklch(0.23 0.035 250)',
          input: 'oklch(0.12 0.02 250)',
        },
        border: {
          subtle: 'oklch(0.26 0.03 250)',
          strong: 'oklch(0.38 0.04 250)',
          focus: 'oklch(0.65 0.15 250)',
        },
        model: {
          lr: 'oklch(0.75 0.18 195)',   // Electric Cyan (Logistic Regression)
          knn: 'oklch(0.72 0.22 325)',  // Vibrant Violet (KNN)
          svm: 'oklch(0.78 0.19 85)',   // Amber Gold (SVM)
          dt: 'oklch(0.76 0.22 145)',   // Emerald Green (Decision Tree)
          nb: 'oklch(0.75 0.22 40)',    // Bright Coral (Naive Bayes)
        },
        status: {
          positive: 'oklch(0.76 0.20 145)',
          negative: 'oklch(0.68 0.18 25)',
          warning: 'oklch(0.78 0.18 85)',
        },
      },
      fontFamily: {
        display: ['var(--font-display)', 'Cabinet Grotesk', 'sans-serif'],
        body: ['var(--font-body)', 'Plus Jakarta Sans', 'sans-serif'],
        mono: ['var(--font-mono)', 'JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        pill: '9999px',
      },
    },
  },
  plugins: [],
};

export default config;
