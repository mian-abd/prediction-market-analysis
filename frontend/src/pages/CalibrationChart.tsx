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
import { Loader2, AlertCircle, Target } from 'lucide-react'
import apiClient from '../api/client'

interface CalibrationPoint {
  market_price: number
  calibrated_price: number
  sample_count: number
  bias: number
}

interface CalibrationData {
  curve: CalibrationPoint[]
  overall_bias: number
  brier_score: number
  total_samples: number
}

export default function CalibrationChart() {
  const { data, isLoading, error } = useQuery<CalibrationData>({
    queryKey: ['calibration-curve'],
    queryFn: async () => {
      const response = await apiClient.get('/calibration/curve')
      return response.data
    },
    refetchInterval: 60_000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-gray-400">
        <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
        <p className="text-lg font-medium text-white mb-2">
          Failed to load calibration data
        </p>
        <p className="text-sm">Check API connection and try again.</p>
      </div>
    )
  }

  const calibration = data!

  const chartData = calibration.curve.map((point) => ({
    market_price: point.market_price * 100,
    calibrated_price: point.calibrated_price * 100,
    perfect: point.market_price * 100,
    bias: point.bias * 100,
    sample_count: point.sample_count,
  }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Calibration Analysis</h1>
        <p className="text-sm text-gray-400 mt-1">
          Market price accuracy vs actual outcomes
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-2">
            <Target className="h-4 w-4 text-blue-400" />
            <p className="text-sm text-gray-400">Overall Bias</p>
          </div>
          <p
            className={`text-2xl font-bold font-mono ${
              Math.abs(calibration.overall_bias) < 0.02
                ? 'text-emerald-400'
                : Math.abs(calibration.overall_bias) < 0.05
                  ? 'text-amber-400'
                  : 'text-red-400'
            }`}
          >
            {calibration.overall_bias > 0 ? '+' : ''}
            {(calibration.overall_bias * 100).toFixed(2)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {Math.abs(calibration.overall_bias) < 0.02
              ? 'Well calibrated'
              : calibration.overall_bias > 0
                ? 'Markets overconfident'
                : 'Markets underconfident'}
          </p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <p className="text-sm text-gray-400 mb-2">Brier Score</p>
          <p className="text-2xl font-bold text-white font-mono">
            {calibration.brier_score.toFixed(4)}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Lower is better (0 = perfect)
          </p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <p className="text-sm text-gray-400 mb-2">Total Samples</p>
          <p className="text-2xl font-bold text-white">
            {calibration.total_samples.toLocaleString()}
          </p>
          <p className="text-xs text-gray-500 mt-1">Resolved markets</p>
        </div>
      </div>

      {/* Calibration Chart */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-1">
          Calibration Curve
        </h2>
        <p className="text-sm text-gray-400 mb-6">
          The diagonal line represents perfect calibration. Areas above the line
          indicate overconfidence; below indicates underconfidence.
        </p>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
            >
              <defs>
                <linearGradient id="biasGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="market_price"
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                tickFormatter={(v: number) => `${v}%`}
                label={{
                  value: 'Market Price',
                  position: 'insideBottom',
                  offset: -5,
                  fill: '#9ca3af',
                  fontSize: 13,
                }}
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                tickFormatter={(v: number) => `${v}%`}
                domain={[0, 100]}
                label={{
                  value: 'Calibrated Price',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 10,
                  fill: '#9ca3af',
                  fontSize: 13,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb',
                }}
                formatter={(value?: number, name?: string) => {
                  const labels: Record<string, string> = {
                    calibrated_price: 'Calibrated',
                    perfect: 'Perfect',
                  }
                  return [`${(value ?? 0).toFixed(1)}%`, labels[name ?? ''] ?? name]
                }}
                labelFormatter={(label) => `Market Price: ${label}%`}
              />
              {/* Shaded area between calibrated and perfect */}
              <Area
                type="monotone"
                dataKey="calibrated_price"
                stroke="none"
                fill="url(#biasGradient)"
              />
              {/* Perfect calibration diagonal */}
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 100, y: 100 },
                ]}
                stroke="#6b7280"
                strokeDasharray="6 4"
                strokeWidth={1.5}
              />
              {/* Actual calibration line */}
              <Line
                type="monotone"
                dataKey="calibrated_price"
                stroke="#3b82f6"
                strokeWidth={2.5}
                dot={{ fill: '#3b82f6', r: 4, stroke: '#1f2937', strokeWidth: 2 }}
                activeDot={{ r: 6 }}
                name="Calibrated"
              />
              {/* Perfect line for legend/tooltip */}
              <Line
                type="monotone"
                dataKey="perfect"
                stroke="#6b7280"
                strokeDasharray="6 4"
                strokeWidth={1.5}
                dot={false}
                name="Perfect"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bias Breakdown */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Bias by Price Bucket
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-left">
                <th className="px-4 py-3 text-gray-400 font-medium">
                  Market Price
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Calibrated Price
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Bias
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Samples
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700/50">
              {calibration.curve.map((point, i) => (
                <tr key={i} className="hover:bg-gray-700/30 transition-colors">
                  <td className="px-4 py-3 font-mono text-white">
                    {(point.market_price * 100).toFixed(0)}%
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-blue-400">
                    {(point.calibrated_price * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span
                      className={`font-mono text-xs px-2 py-0.5 rounded ${
                        Math.abs(point.bias) < 0.02
                          ? 'bg-emerald-900/30 text-emerald-400'
                          : Math.abs(point.bias) < 0.05
                            ? 'bg-amber-900/30 text-amber-400'
                            : 'bg-red-900/30 text-red-400'
                      }`}
                    >
                      {point.bias > 0 ? '+' : ''}
                      {(point.bias * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-gray-400">
                    {point.sample_count.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
