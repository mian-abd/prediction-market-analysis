/**
 * Cross-Platform Correlation Matrix
 *
 * Shows price correlations between markets in a heatmap.
 * Use cases:
 * - Find related markets for hedging
 * - Identify anomalies (high correlation + price divergence)
 * - Discover market clusters
 */

import { useEffect, useState } from 'react'
import { Loader2, Network } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'
import ErrorState from '../ErrorState'

interface CorrelationPair {
  market_a_id: number
  market_a_question: string
  market_a_platform: string
  market_b_id: number
  market_b_question: string
  market_b_platform: string
  correlation: number
}

interface CorrelationData {
  markets: Array<{
    id: number
    question: string
    platform: string
    category: string | null
  }>
  correlations: CorrelationPair[]
  lookback_days: number
  min_correlation: number
  total_pairs: number
  message?: string
}

interface CorrelationMatrixProps {
  category?: string
  minCorrelation?: number
  lookbackDays?: number
}

export default function CorrelationMatrix({
  category,
  minCorrelation = 0.3,
  lookbackDays = 7,
}: CorrelationMatrixProps) {
  const [data, setData] = useState<CorrelationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCorrelations = async () => {
    setLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams({
        min_correlation: minCorrelation.toString(),
        lookback_days: lookbackDays.toString(),
      })
      if (category) {
        params.append('category', category)
      }

      const response = await apiClient.get(`/analytics/correlations?${params}`)
      setData(response.data)
    } catch (err: any) {
      setError('Failed to load correlations')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCorrelations()
  }, [category, minCorrelation, lookbackDays])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Failed to load correlations"
        message={error || 'Could not compute market correlations.'}
        onRetry={fetchCorrelations}
        showBackendHint={false}
      />
    )
  }

  if (data.correlations.length === 0) {
    return (
      <EmptyState
        icon={Network}
        title={data.message || 'No correlations found'}
        message="Try lowering the minimum correlation threshold or select a different category."
      />
    )
  }

  // Get color based on correlation value
  const getCorrelationColor = (corr: number): string => {
    if (corr > 0.7) return '#4CAF70' // Strong positive (green)
    if (corr > 0.4) return '#A3D97A' // Moderate positive (light green)
    if (corr > 0) return '#E8F5E9' // Weak positive (very light green)
    if (corr > -0.4) return '#FFEBEE' // Weak negative (very light red)
    if (corr > -0.7) return '#EF9A9A' // Moderate negative (light red)
    return '#CF6679' // Strong negative (red)
  }

  const getTextColor = (corr: number): string => {
    // Use dark text for light backgrounds
    if (corr >= -0.4 && corr <= 0.4) return '#1A1A1C'
    return '#FFFFFF'
  }

  return (
    <div className="space-y-6">
      {/* Stats header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Total Pairs
            </p>
            <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
              {data.total_pairs}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Lookback
            </p>
            <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
              {data.lookback_days}d
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Min Correlation
            </p>
            <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
              {(data.min_correlation * 100).toFixed(0)}%
            </p>
          </div>
        </div>

        {/* Color scale legend */}
        <div className="flex items-center gap-2 text-[11px]">
          <span style={{ color: 'var(--text-3)' }}>Correlation:</span>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 rounded" style={{ background: '#CF6679' }} />
            <span style={{ color: 'var(--text-3)' }}>-1</span>
          </div>
          <div className="w-6 h-4 rounded" style={{ background: '#E8F5E9' }} />
          <span style={{ color: 'var(--text-3)' }}>0</span>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 rounded" style={{ background: '#4CAF70' }} />
            <span style={{ color: 'var(--text-3)' }}>+1</span>
          </div>
        </div>
      </div>

      {/* Correlation pairs list */}
      <div className="space-y-2">
        <p className="text-[13px] font-medium mb-3" style={{ color: 'var(--text)' }}>
          Top Correlated Pairs
        </p>

        <div className="space-y-2">
          {data.correlations.slice(0, 20).map((pair, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 p-3 rounded-xl"
              style={{ background: 'var(--card)' }}
            >
              {/* Correlation badge */}
              <div
                className="px-3 py-2 rounded-lg font-mono font-semibold text-[14px] flex-shrink-0"
                style={{
                  background: getCorrelationColor(pair.correlation),
                  color: getTextColor(pair.correlation),
                }}
              >
                {(pair.correlation * 100).toFixed(0)}%
              </div>

              {/* Market A */}
              <div className="flex-1 min-w-0">
                <p className="text-[12px] truncate" style={{ color: 'var(--text)' }}>
                  {pair.market_a_question}
                </p>
                <p className="text-[10px] uppercase mt-0.5" style={{ color: 'var(--text-3)' }}>
                  {pair.market_a_platform}
                </p>
              </div>

              {/* Arrow */}
              <div className="flex-shrink-0">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M7 14L13 10L7 6"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{ color: 'var(--text-3)' }}
                  />
                </svg>
              </div>

              {/* Market B */}
              <div className="flex-1 min-w-0">
                <p className="text-[12px] truncate" style={{ color: 'var(--text)' }}>
                  {pair.market_b_question}
                </p>
                <p className="text-[10px] uppercase mt-0.5" style={{ color: 'var(--text-3)' }}>
                  {pair.market_b_platform}
                </p>
              </div>
            </div>
          ))}
        </div>

        {data.correlations.length > 20 && (
          <p className="text-[12px] text-center pt-3" style={{ color: 'var(--text-3)' }}>
            Showing top 20 of {data.total_pairs} pairs
          </p>
        )}
      </div>
    </div>
  )
}
