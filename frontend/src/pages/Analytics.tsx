/**
 * Analytics Page
 *
 * Advanced market analytics including:
 * - Cross-platform correlation matrix
 * - Market cluster analysis
 * - Anomaly detection (future)
 */

import { useState } from 'react'
import { Brain } from 'lucide-react'
import CorrelationMatrix from '../components/charts/CorrelationMatrix'

export default function Analytics() {
  const [category, setCategory] = useState<string>('')
  const [minCorrelation, setMinCorrelation] = useState(0.3)
  const [lookbackDays, setLookbackDays] = useState(7)

  return (
    <div className="space-y-6 fade-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Brain className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-[22px] font-bold" style={{ color: 'var(--text)' }}>
              Analytics
            </h1>
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              Market correlations and advanced insights
            </p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="card p-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Category filter */}
          <div>
            <label className="text-[11px] uppercase font-medium mb-2 block" style={{ color: 'var(--text-3)' }}>
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full px-3 py-2 rounded-lg text-[13px] border"
              style={{
                background: 'var(--bg)',
                color: 'var(--text)',
                borderColor: 'var(--border)',
              }}
            >
              <option value="">All Categories</option>
              <option value="politics">Politics</option>
              <option value="crypto">Crypto</option>
              <option value="sports">Sports</option>
              <option value="business">Business</option>
              <option value="science">Science</option>
            </select>
          </div>

          {/* Min correlation */}
          <div>
            <label className="text-[11px] uppercase font-medium mb-2 block" style={{ color: 'var(--text-3)' }}>
              Min Correlation: {(minCorrelation * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minCorrelation}
              onChange={(e) => setMinCorrelation(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Lookback days */}
          <div>
            <label className="text-[11px] uppercase font-medium mb-2 block" style={{ color: 'var(--text-3)' }}>
              Lookback: {lookbackDays} days
            </label>
            <input
              type="range"
              min="1"
              max="30"
              step="1"
              value={lookbackDays}
              onChange={(e) => setLookbackDays(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Correlation Matrix */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Cross-Platform Correlation Matrix
        </h2>
        <CorrelationMatrix
          category={category || undefined}
          minCorrelation={minCorrelation}
          lookbackDays={lookbackDays}
        />
      </div>
    </div>
  )
}
