/**
 * Correlation Explorer
 *
 * Two complementary modes:
 * - List Mode: Ranked pairs with correlation bars and filtering
 * - Graph Mode: Interactive force-directed network visualization
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Network, List, Share2 } from 'lucide-react'
import apiClient from '../api/client'
import CorrelationMatrix from '../components/charts/CorrelationMatrix'
import CorrelationGraph from '../components/charts/CorrelationGraph'
import { Skeleton } from '../components/LoadingSkeleton'

type ViewMode = 'list' | 'graph'

interface CorrelationData {
  markets: Array<{
    id: number
    question: string
    platform: string
    category: string | null
  }>
  correlations: Array<{
    market_a_id: number
    market_a_question: string
    market_a_platform: string
    market_b_id: number
    market_b_question: string
    market_b_platform: string
    correlation: number
  }>
  lookback_days: number
  min_correlation: number
  total_pairs: number
  message?: string
}

export default function Analytics() {
  const [viewMode, setViewMode] = useState<ViewMode>('graph')
  const [category, setCategory] = useState<string>('')
  const [minCorrelation, setMinCorrelation] = useState(0.3)
  const [lookbackDays, setLookbackDays] = useState(7)

  // Pre-fetch data for graph mode (CorrelationMatrix fetches its own)
  const { data: graphData, isLoading: graphLoading } = useQuery<CorrelationData>({
    queryKey: ['correlations-graph', category, minCorrelation, lookbackDays],
    queryFn: async () => {
      const params = new URLSearchParams({
        min_correlation: minCorrelation.toString(),
        lookback_days: lookbackDays.toString(),
      })
      if (category) params.append('category', category)
      const response = await apiClient.get(`/analytics/correlations?${params}`)
      return response.data
    },
    enabled: viewMode === 'graph',
    staleTime: 120_000,
    retry: 1,
  })

  return (
    <div className="space-y-5 fade-up">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Network className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-[22px] font-bold" style={{ color: 'var(--text)' }}>
              Correlation Explorer
            </h1>
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              Identify correlated markets and cluster risk
            </p>
          </div>
        </div>

        {/* Mode Toggle */}
        <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--card)', border: '1px solid var(--border)' }}>
          <button
            onClick={() => setViewMode('list')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors"
            style={{
              background: viewMode === 'list' ? 'var(--accent-dim)' : 'transparent',
              color: viewMode === 'list' ? 'var(--accent)' : 'var(--text-3)',
            }}
          >
            <List className="h-3.5 w-3.5" />
            List
          </button>
          <button
            onClick={() => setViewMode('graph')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors"
            style={{
              background: viewMode === 'graph' ? 'var(--accent-dim)' : 'transparent',
              color: viewMode === 'graph' ? 'var(--accent)' : 'var(--text-3)',
            }}
          >
            <Share2 className="h-3.5 w-3.5" />
            Graph
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="card p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Category filter */}
          <div>
            <label className="text-[10px] uppercase font-semibold mb-1.5 block" style={{ color: 'var(--text-3)' }}>
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="input"
              style={{ padding: '8px 12px', fontSize: '13px' }}
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
            <label className="text-[10px] uppercase font-semibold mb-1.5 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Min Correlation</span>
              <span className="text-[12px] font-mono font-bold" style={{ color: 'var(--accent)' }}>
                {(minCorrelation * 100).toFixed(0)}%
              </span>
            </label>
            <input
              type="range"
              min="0"
              max="0.95"
              step="0.05"
              value={minCorrelation}
              onChange={(e) => setMinCorrelation(parseFloat(e.target.value))}
              className="w-full"
              style={{ accentColor: '#C4A24D' }}
            />
          </div>

          {/* Lookback days */}
          <div>
            <label className="text-[10px] uppercase font-semibold mb-1.5 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Lookback</span>
              <span className="text-[12px] font-mono font-bold" style={{ color: 'var(--accent)' }}>
                {lookbackDays}d
              </span>
            </label>
            <input
              type="range"
              min="1"
              max="365"
              step="1"
              value={lookbackDays}
              onChange={(e) => setLookbackDays(parseInt(e.target.value))}
              className="w-full"
              style={{ accentColor: '#C4A24D' }}
            />
          </div>
        </div>
      </div>

      {/* Content */}
      {viewMode === 'list' ? (
        <div className="card p-5">
          <h2 className="text-[15px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
            Top Correlated Pairs
          </h2>
          <CorrelationMatrix
            category={category || undefined}
            minCorrelation={minCorrelation}
            lookbackDays={lookbackDays}
          />
        </div>
      ) : (
        <div className="card overflow-hidden" style={{ minHeight: '500px' }}>
          {graphLoading ? (
            <div className="flex items-center justify-center h-[500px]">
              <div className="text-center space-y-3">
                <Skeleton className="h-12 w-12 rounded-full mx-auto" />
                <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>Computing correlations...</p>
                <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>This may take a few seconds</p>
              </div>
            </div>
          ) : graphData && graphData.markets.length >= 2 ? (
            <CorrelationGraph data={graphData} />
          ) : (
            <div className="flex flex-col items-center justify-center h-[500px] gap-3">
              <Network className="h-6 w-6" style={{ color: 'var(--text-3)' }} />
              <p className="text-[14px] font-medium" style={{ color: 'var(--text-2)' }}>
                {graphData?.message || 'Not enough data for graph'}
              </p>
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                Try lowering the minimum correlation or selecting a specific category
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
