import { useQuery } from '@tanstack/react-query'
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts'
import { Loader2, AlertCircle } from 'lucide-react'
import apiClient from '../api/client'

interface CalibrationPoint {
  market_price: number
  calibrated_price: number
  sample_count: number
  bias: number
}

export default function CalibrationChart() {
  const { data, isLoading, error } = useQuery<{ curve: CalibrationPoint[] }>({
    queryKey: ['calibration-curve'],
    queryFn: async () => {
      const response = await apiClient.get('/calibration/curve')
      return response.data
    },
    refetchInterval: 60_000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-80 gap-3">
        <AlertCircle className="h-8 w-8" style={{ color: 'var(--red)' }} />
        <p className="text-[14px] font-medium">Failed to load calibration data</p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Check API connection.</p>
      </div>
    )
  }

  const curve = data?.curve ?? []
  const overallBias = curve.length > 0 ? curve.reduce((s, p) => s + p.bias, 0) / curve.length : 0
  const totalSamples = curve.reduce((s, p) => s + p.sample_count, 0)

  const chartData = curve.map((point) => ({
    market_price: point.market_price * 100,
    calibrated_price: point.calibrated_price * 100,
    perfect: point.market_price * 100,
    bias: point.bias * 100,
    sample_count: point.sample_count,
  }))

  return (
    <div className="space-y-8 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>
          Calibration Analysis
        </h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Market price accuracy vs actual outcomes
        </p>
      </div>

      {/* Summary */}
      <div className="card p-6">
        <div className="grid grid-cols-3 gap-6">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Overall Bias
            </p>
            <p
              className="text-[24px] font-bold font-mono"
              style={{
                color: Math.abs(overallBias) < 0.02 ? 'var(--green)' :
                  Math.abs(overallBias) < 0.05 ? 'var(--accent)' : 'var(--red)',
              }}
            >
              {overallBias > 0 ? '+' : ''}{(overallBias * 100).toFixed(2)}%
            </p>
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>
              {Math.abs(overallBias) < 0.02 ? 'Well calibrated' : overallBias > 0 ? 'Overconfident' : 'Underconfident'}
            </p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Calibration Points
            </p>
            <p className="text-[24px] font-bold">{curve.length}</p>
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>Price buckets</p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Total Samples
            </p>
            <p className="text-[24px] font-bold">{totalSamples.toLocaleString()}</p>
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>Resolved markets</p>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="card p-6">
        <p className="text-[14px] font-semibold mb-1" style={{ color: 'var(--text)' }}>
          Calibration Curve
        </p>
        <p className="text-[12px] mb-6" style={{ color: 'var(--text-3)' }}>
          Diagonal = perfect calibration. Above = overconfidence.
        </p>
        <div className="h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
              <defs>
                <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#C4A24D" stopOpacity={0.08} />
                  <stop offset="95%" stopColor="#C4A24D" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="market_price"
                stroke="rgba(255,255,255,0.06)"
                tick={{ fill: '#48484A', fontSize: 11 }}
                tickFormatter={(v: number) => `${v}%`}
                label={{ value: 'Market Price', position: 'insideBottom', offset: -5, fill: '#48484A', fontSize: 11 }}
              />
              <YAxis
                stroke="rgba(255,255,255,0.06)"
                tick={{ fill: '#48484A', fontSize: 11 }}
                tickFormatter={(v: number) => `${v}%`}
                domain={[0, 100]}
                label={{ value: 'Calibrated', angle: -90, position: 'insideLeft', offset: 10, fill: '#48484A', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1A1A1C',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '12px',
                  color: '#FFFFFF',
                  fontSize: '12px',
                  padding: '10px 14px',
                }}
                formatter={(value?: number, name?: string) => {
                  const labels: Record<string, string> = { calibrated_price: 'Calibrated', perfect: 'Perfect' }
                  return [`${(value ?? 0).toFixed(1)}%`, labels[name ?? ''] ?? name]
                }}
                labelFormatter={(label) => `Market: ${label}%`}
              />
              <Area type="monotone" dataKey="calibrated_price" stroke="none" fill="url(#areaFill)" />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                stroke="rgba(255,255,255,0.08)"
                strokeDasharray="6 4"
                strokeWidth={1.5}
              />
              <Line
                type="monotone"
                dataKey="calibrated_price"
                stroke="#C4A24D"
                strokeWidth={2}
                dot={{ fill: '#C4A24D', r: 3.5, stroke: '#000', strokeWidth: 2 }}
                activeDot={{ r: 5.5, fill: '#C4A24D', stroke: '#000', strokeWidth: 2 }}
                name="Calibrated"
              />
              <Line
                type="monotone"
                dataKey="perfect"
                stroke="rgba(255,255,255,0.08)"
                strokeDasharray="6 4"
                strokeWidth={1.5}
                dot={false}
                name="Perfect"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bias Rows */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Bias by Price Bucket
        </p>
        <div className="space-y-1">
          {curve.map((point, i) => {
            const biasAbs = Math.abs(point.bias)
            const color = biasAbs < 0.02 ? 'var(--green)' : biasAbs < 0.05 ? 'var(--accent)' : 'var(--red)'
            return (
              <div
                key={i}
                className="card flex items-center gap-4 px-5 py-3"
              >
                <span className="w-12 text-[13px] font-mono font-medium" style={{ color: 'var(--text)' }}>
                  {(point.market_price * 100).toFixed(0)}%
                </span>
                <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.04)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min(point.calibrated_price * 100, 100)}%`,
                      background: color,
                      opacity: 0.6,
                    }}
                  />
                </div>
                <span className="w-14 text-right text-[12px] font-mono" style={{ color: 'var(--blue)' }}>
                  {(point.calibrated_price * 100).toFixed(1)}%
                </span>
                <span
                  className="w-16 text-right text-[12px] font-mono font-medium"
                  style={{ color }}
                >
                  {point.bias > 0 ? '+' : ''}{(point.bias * 100).toFixed(2)}%
                </span>
                <span className="w-10 text-right text-[11px]" style={{ color: 'var(--text-3)' }}>
                  {point.sample_count}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
