import { useEffect, useRef, useState } from 'react'
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts'
import type { IChartApi, ISeriesApi, CandlestickData, Time, LineData, HistogramData } from 'lightweight-charts'

export type ChartType = 'candlestick' | 'line' | 'area'
export type TimeInterval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

interface PriceDataPoint {
  timestamp: number // Unix timestamp in seconds
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface PriceChartProps {
  marketId: number
  interval?: TimeInterval
  type?: ChartType
  height?: number
  showVolume?: boolean
  showCrosshair?: boolean
  autoRefresh?: boolean // Auto-refresh every 60 seconds
  data?: PriceDataPoint[] // Optional: provide data directly (for testing)
}

export default function PriceChart({
  marketId,
  interval = '5m',
  type = 'candlestick',
  height = 400,
  showVolume = true,
  showCrosshair = true,
  autoRefresh = true,
  data: providedData,
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const mainSeriesRef = useRef<ISeriesApi<'Candlestick' | 'Line' | 'Area'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [priceData, setPriceData] = useState<PriceDataPoint[]>([])

  // Fetch price data from API
  useEffect(() => {
    if (providedData) {
      setPriceData(providedData)
      setIsLoading(false)
      return
    }

    const fetchPriceData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // Fetch price snapshots for this market
        const response = await fetch(
          `/api/v1/markets/${marketId}/price-history?interval=${interval}&limit=500`
        )

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const result = await response.json()
        setPriceData(result.data || [])
      } catch (err) {
        console.error('Failed to fetch price data:', err)
        setError(err instanceof Error ? err.message : 'Failed to load chart data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchPriceData()

    // Auto-refresh if enabled
    if (autoRefresh) {
      const intervalId = setInterval(fetchPriceData, 60000) // Every 60 seconds
      return () => clearInterval(intervalId)
    }
  }, [marketId, interval, autoRefresh, providedData])

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    // Create chart instance
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: showVolume ? height : height - 100,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9CA3AF', // Tailwind gray-400
      },
      grid: {
        vertLines: { color: '#1F2937' }, // Tailwind gray-800
        horzLines: { color: '#1F2937' },
      },
      crosshair: {
        mode: showCrosshair ? CrosshairMode.Normal : CrosshairMode.Hidden,
        vertLine: {
          width: 1,
          color: '#4B5563',
          style: 0,
          labelBackgroundColor: '#C4A24D', // Accent color
        },
        horzLine: {
          width: 1,
          color: '#4B5563',
          style: 0,
          labelBackgroundColor: '#C4A24D',
        },
      },
      rightPriceScale: {
        borderColor: '#374151',
        scaleMargins: {
          top: 0.1,
          bottom: showVolume ? 0.3 : 0.1,
        },
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: interval === '1m',
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    })

    chartRef.current = chart

    // Create main price series based on type
    if (type === 'candlestick') {
      mainSeriesRef.current = chart.addCandlestickSeries({
        upColor: '#10B981', // Tailwind green-500
        downColor: '#EF4444', // Tailwind red-500
        borderUpColor: '#10B981',
        borderDownColor: '#EF4444',
        wickUpColor: '#10B981',
        wickDownColor: '#EF4444',
      })
    } else if (type === 'line') {
      mainSeriesRef.current = chart.addLineSeries({
        color: '#C4A24D', // Accent color
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: '#C4A24D',
        crosshairMarkerBackgroundColor: '#000000',
      })
    } else if (type === 'area') {
      mainSeriesRef.current = chart.addAreaSeries({
        topColor: 'rgba(196, 162, 77, 0.4)', // Accent with alpha
        bottomColor: 'rgba(196, 162, 77, 0.0)',
        lineColor: '#C4A24D',
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
      })
    }

    // Create volume series if enabled
    if (showVolume) {
      volumeSeriesRef.current = chart.addHistogramSeries({
        color: '#4B5563', // Tailwind gray-600
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '', // Use separate price scale
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })
    }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [type, height, showVolume, showCrosshair, interval])

  // Update chart data
  useEffect(() => {
    if (!mainSeriesRef.current || !priceData.length) return

    try {
      if (type === 'candlestick') {
        const candlestickData: CandlestickData<Time>[] = priceData.map((d) => ({
          time: d.timestamp as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }))
        mainSeriesRef.current.setData(candlestickData)
      } else {
        // For line and area, use close price
        const lineData: LineData<Time>[] = priceData.map((d) => ({
          time: d.timestamp as Time,
          value: d.close,
        }))
        mainSeriesRef.current.setData(lineData)
      }

      // Update volume data
      if (showVolume && volumeSeriesRef.current) {
        const volumeData: HistogramData<Time>[] = priceData.map((d, i) => {
          const prevClose = i > 0 ? priceData[i - 1].close : d.open
          const isUp = d.close >= prevClose

          return {
            time: d.timestamp as Time,
            value: d.volume,
            color: isUp ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)', // Green/Red with alpha
          }
        })
        volumeSeriesRef.current.setData(volumeData)
      }

      // Fit content to visible range
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent()
      }
    } catch (err) {
      console.error('Failed to update chart data:', err)
    }
  }, [priceData, type, showVolume])

  if (isLoading) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <div
            className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"
            style={{ color: 'var(--accent)' }}
          />
          <p className="mt-4 text-sm" style={{ color: 'var(--text-3)' }}>
            Loading chart data...
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <p className="text-sm" style={{ color: 'var(--red)' }}>
            Failed to load chart
          </p>
          <p className="mt-2 text-xs" style={{ color: 'var(--text-3)' }}>
            {error}
          </p>
        </div>
      </div>
    )
  }

  if (!priceData.length) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            No price data available
          </p>
          <p className="mt-2 text-xs" style={{ color: 'var(--text-3)' }}>
            Try selecting a different interval
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      style={{
        background: 'var(--card)',
        borderRadius: '12px',
        padding: '16px',
        border: '1px solid var(--border)',
      }}
    >
      <div ref={chartContainerRef} style={{ height: `${height}px` }} />
    </div>
  )
}
