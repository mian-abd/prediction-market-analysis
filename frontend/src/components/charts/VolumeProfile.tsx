/**
 * Volume Profile Histogram
 *
 * Shows distribution of trading volume by price level:
 * - Horizontal bars (price on Y-axis, volume on X-axis)
 * - Point of Control (POC) - highest volume price
 * - Value Area (70% of volume) highlighted
 * - Useful for identifying support/resistance levels
 */

import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import { Loader2, AlertCircle, Target } from 'lucide-react'
import apiClient from '../../api/client'

interface VolumeBucket {
  price_level: number
  volume: number
  is_poc: boolean
  is_value_area: boolean
}

interface VolumeProfileProps {
  marketId: number
  bucketSize?: number // Price bucket size (e.g., 0.01)
  lookbackDays?: number
}

export default function VolumeProfile({
  marketId,
  bucketSize = 0.01,
  lookbackDays = 7,
}: VolumeProfileProps) {
  const [data, setData] = useState<VolumeBucket[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [poc, setPoc] = useState<number | null>(null)

  const fetchVolumeProfile = async () => {
    setLoading(true)
    setError(null)

    try {
      // Fetch price history
      const response = await apiClient.get(
        `/markets/${marketId}/price-history?interval=1h&limit=${lookbackDays * 24}`
      )
      const priceData = response.data.data

      if (!priceData || priceData.length === 0) {
        setData([])
        return
      }

      // Group by price buckets
      const buckets = new Map<number, number>()

      priceData.forEach((candle: any) => {
        // Use average price (OHLC/4)
        const avgPrice = (candle.open + candle.high + candle.low + candle.close) / 4
        const bucket = Math.floor(avgPrice / bucketSize) * bucketSize
        const volume = candle.volume || 0

        buckets.set(bucket, (buckets.get(bucket) || 0) + volume)
      })

      // Convert to array and sort by price
      const bucketArray = Array.from(buckets.entries())
        .map(([price, volume]) => ({
          price_level: price,
          volume,
          is_poc: false,
          is_value_area: false,
        }))
        .sort((a, b) => a.price_level - b.price_level)

      if (bucketArray.length === 0) {
        setData([])
        return
      }

      // Find Point of Control (highest volume)
      const pocBucket = bucketArray.reduce((max, bucket) =>
        bucket.volume > max.volume ? bucket : max
      )
      pocBucket.is_poc = true
      setPoc(pocBucket.price_level)

      // Calculate Value Area (70% of total volume)
      const totalVolume = bucketArray.reduce((sum, b) => sum + b.volume, 0)
      const targetVolume = totalVolume * 0.7

      // Sort by volume to find value area
      const sortedByVolume = [...bucketArray].sort((a, b) => b.volume - a.volume)
      let cumulativeVolume = 0
      sortedByVolume.forEach((bucket) => {
        if (cumulativeVolume < targetVolume) {
          bucket.is_value_area = true
          cumulativeVolume += bucket.volume
        }
      })

      setData(bucketArray)
    } catch (err: any) {
      setError('Failed to load volume profile')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchVolumeProfile()
  }, [marketId, bucketSize, lookbackDays])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-5 w-5 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2">
        <AlertCircle className="h-5 w-5" style={{ color: 'var(--red)' }} />
        <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
          {error || 'No volume profile data'}
        </p>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2">
        <AlertCircle className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
        <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
          No volume data available
        </p>
      </div>
    )
  }

  // Find max volume for scaling
  const maxVolume = Math.max(...data.map((d) => d.volume))

  return (
    <div className="space-y-4">
      {/* Header with POC */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Target className="h-4 w-4" style={{ color: 'var(--accent)' }} />
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Point of Control
            </p>
            <p className="text-[16px] font-mono font-bold" style={{ color: 'var(--accent)' }}>
              {poc ? poc.toFixed(3) : 'N/A'}
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px]">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: 'var(--accent)' }} />
            <span style={{ color: 'var(--text-3)' }}>POC</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: 'rgba(94,180,239,0.5)' }} />
            <span style={{ color: 'var(--text-3)' }}>Value Area</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: 'rgba(142,142,147,0.3)' }} />
            <span style={{ color: 'var(--text-3)' }}>Other</span>
          </div>
        </div>
      </div>

      {/* Volume profile chart (horizontal bars) */}
      <div style={{ width: '100%', height: '384px', minHeight: '384px' }}>
        <ResponsiveContainer width="100%" height={384}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 10, right: 10, left: 40, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />

            <XAxis
              type="number"
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              tickFormatter={(value) => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value)}
            />

            <YAxis
              type="number"
              dataKey="price_level"
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              tickFormatter={(value) => value.toFixed(3)}
              domain={['dataMin', 'dataMax']}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '8px 12px',
              }}
              formatter={(value: number, name: string, props: any) => {
                const { payload } = props
                return [
                  <div key={name} className="space-y-1">
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Price:</span>
                      <span className="font-mono">{payload.price_level.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Volume:</span>
                      <span className="font-mono">{value.toLocaleString()}</span>
                    </div>
                    {payload.is_poc && (
                      <div className="pt-1 border-t" style={{ borderColor: 'var(--border)' }}>
                        <span style={{ color: 'var(--accent)' }}>Point of Control</span>
                      </div>
                    )}
                    {payload.is_value_area && !payload.is_poc && (
                      <div className="pt-1 border-t" style={{ borderColor: 'var(--border)' }}>
                        <span style={{ color: 'var(--blue)' }}>Value Area</span>
                      </div>
                    )}
                  </div>,
                ]
              }}
            />

            {/* POC reference line */}
            {poc !== null && (
              <ReferenceLine
                y={poc}
                stroke="var(--accent)"
                strokeDasharray="3 3"
                strokeWidth={2}
              />
            )}

            {/* Volume bars */}
            <Bar dataKey="volume" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    entry.is_poc
                      ? 'var(--accent)'
                      : entry.is_value_area
                      ? 'rgba(94,180,239,0.5)'
                      : 'rgba(142,142,147,0.3)'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
