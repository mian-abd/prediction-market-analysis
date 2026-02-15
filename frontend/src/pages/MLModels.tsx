import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Brain,
  TrendingUp,
  TrendingDown,
  ChevronRight,
  Layers,
  BarChart3,
  Target,
} from 'lucide-react'
import {
  ComposedChart,
  Scatter,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import apiClient from '../api/client'
import SignalAccuracyChart from '../components/charts/SignalAccuracyChart'

interface MispricedMarket {
  market_id: number
  question: string
  category: string | null
  price_yes: number
  calibrated_price: number
  delta_pct: number
  direction: string
  edge_estimate: number
  volume_24h: number | null
}

interface AccuracyData {
  trained: boolean
  metrics: {
    n_total_resolved: number
    n_usable: number
    n_train: number
    n_test: number
    class_balance_yes_pct: number
    baseline_brier: number
    calibration_brier: number
    ensemble_brier: number
    ensemble_auc: number
    temporal_split_date?: string
    oof_brier?: {
      calibration: number
      xgboost: number
      lightgbm: number
    }
    xgb_feature_importance: Record<string, number>
    lgb_feature_importance: Record<string, number>
  }
  weights: Record<string, number>
  models?: Record<string, {
    name: string
    type: string
    brier_score: number | null
    weight: number
    feature_importance?: Record<string, number>
  }>
  baseline_brier: number
  training_samples: number
  test_samples: number
  total_resolved: number
  usable_samples: number
  profit_simulation?: {
    gated_pnl: number
    gated_win_rate: number
    gated_trades: number
    ungated_pnl: number
    ungated_trades: number
    ungated_win_rate: number
    min_net_edge: number
  }
  ablation?: {
    calibration_only: number
    cal_plus_xgb: number
    cal_plus_lgb: number
    full_ensemble: number
  }
  error?: string
}

export default function MLModels() {
  const navigate = useNavigate()
  const [directionFilter, setDirectionFilter] = useState<'all' | 'overpriced' | 'underpriced'>('all')
  const [minEdge, setMinEdge] = useState(0)

  const { data, isLoading, error } = useQuery<{ markets: MispricedMarket[] }>({
    queryKey: ['top-mispriced'],
    queryFn: async () => {
      const response = await apiClient.get('/predictions/top/mispriced', { params: { limit: 30 } })
      return response.data
    },
    refetchInterval: 60_000,
  })

  const { data: accuracy } = useQuery<AccuracyData>({
    queryKey: ['model-accuracy'],
    queryFn: async () => {
      const response = await apiClient.get('/predictions/accuracy')
      return response.data
    },
    refetchInterval: 300_000,
  })

  const { data: calCurveData } = useQuery<{ curve: Array<{ market_price: number; calibrated_price: number }> }>({
    queryKey: ['calibration-curve'],
    queryFn: async () => (await apiClient.get('/calibration/curve')).data,
    refetchInterval: 600_000,
    retry: 1,
  })

  const markets = data?.markets ?? []

  const filteredMarkets = useMemo(() => {
    let result = [...markets]
    if (directionFilter !== 'all') {
      result = result.filter((m) => m.direction === directionFilter)
    }
    if (minEdge > 0) {
      result = result.filter((m) => m.edge_estimate * 100 >= minEdge)
    }
    return result
  }, [markets, directionFilter, minEdge])

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
        <p className="text-[14px] font-medium">Failed to load ML predictions</p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Check API connection.</p>
      </div>
    )
  }

  const overCount = markets.filter((m) => m.direction === 'overpriced').length
  const underCount = markets.filter((m) => m.direction === 'underpriced').length
  const avgEdge = markets.length > 0
    ? markets.reduce((s, m) => s + (m.edge_estimate || 0), 0) / markets.length * 100
    : 0

  const brierImprovement = accuracy?.trained && accuracy.metrics?.ensemble_brier && accuracy.metrics?.baseline_brier
    ? ((1 - accuracy.metrics.ensemble_brier / accuracy.metrics.baseline_brier) * 100)
    : null

  return (
    <div className="space-y-6 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>ML Models</h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Ensemble predictions and mispriced market detection
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Markets Analyzed', value: markets.length.toString(), color: 'var(--text)' },
          { label: 'Avg Edge', value: `${avgEdge.toFixed(2)}%`, color: 'var(--accent)' },
          { label: 'Overpriced', value: overCount.toString(), color: 'var(--red)' },
          { label: 'Underpriced', value: underCount.toString(), color: 'var(--green)' },
        ].map((s) => (
          <div key={s.label} className="card p-4 text-center">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
            <p className="text-[22px] font-bold" style={{ color: s.color }}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Calibration Model Card */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'var(--accent-dim)' }}
            >
              <Brain className="h-[17px] w-[17px]" style={{ color: 'var(--accent)' }} />
            </div>
            <div className="flex-1">
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                Calibration Model
              </p>
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                Isotonic Regression
              </p>
            </div>
            <span className="pill pill-green">Active</span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Type', value: 'Isotonic' },
              { label: 'Brier Score', value: (accuracy?.models?.calibration?.brier_score != null ? accuracy.models.calibration.brier_score.toFixed(4) : (accuracy?.metrics?.calibration_brier != null ? accuracy.metrics.calibration_brier.toFixed(4) : 'N/A')), color: 'var(--accent)' },
              { label: 'Weight', value: accuracy?.weights?.calibration != null ? `${(accuracy.weights.calibration * 100).toFixed(0)}%` : 'N/A', color: 'var(--blue)' },
            ].map((s) => (
              <div key={s.label} className="text-center py-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)' }}>
                <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
                <p className="text-[14px] font-bold" style={{ color: s.color ?? 'var(--text)' }}>{s.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Ensemble Model Card */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: 'rgba(59,130,246,0.1)' }}
            >
              <Layers className="h-[17px] w-[17px]" style={{ color: 'var(--blue)' }} />
            </div>
            <div className="flex-1">
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                Ensemble Model
              </p>
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                XGBoost + LightGBM + Calibration
              </p>
            </div>
            <span className={`pill ${accuracy?.trained ? 'pill-green' : 'pill-red'}`}>
              {accuracy?.trained ? 'Active' : 'Not Trained'}
            </span>
          </div>
          {accuracy?.trained && accuracy.metrics ? (
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Ensemble Brier', value: accuracy.metrics.ensemble_brier?.toFixed(4) ?? 'N/A', color: 'var(--accent)' },
                { label: 'AUC-ROC', value: accuracy.metrics.ensemble_auc?.toFixed(3) ?? 'N/A', color: 'var(--green)' },
                { label: 'Training Set', value: `${accuracy.metrics.n_usable ?? 0}`, color: 'var(--text)' },
              ].map((s) => (
                <div key={s.label} className="text-center py-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
                  <p className="text-[14px] font-bold" style={{ color: s.color }}>{s.value}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
              Run training script to enable ensemble predictions
            </p>
          )}
        </div>
      </div>

      {/* Calibration Curve */}
      {calCurveData?.curve && calCurveData.curve.length > 0 && (
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <Target className="h-5 w-5" style={{ color: 'var(--accent)' }} />
            <div>
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                Calibration Curve
              </p>
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                Market price vs calibrated probability &middot; Diagonal = perfectly calibrated
              </p>
            </div>
          </div>
          <div style={{ width: '100%', height: '300px' }}>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart
                data={[
                  ...calCurveData.curve.map((p) => ({ ...p, diagonal: p.market_price })),
                  // Ensure diagonal endpoints exist
                  ...(calCurveData.curve[0]?.market_price > 0.01 ? [{ market_price: 0, calibrated_price: null, diagonal: 0 }] : []),
                  ...(calCurveData.curve[calCurveData.curve.length - 1]?.market_price < 0.99 ? [{ market_price: 1, calibrated_price: null, diagonal: 1 }] : []),
                ].sort((a, b) => a.market_price - b.market_price)}
                margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="market_price"
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  stroke="rgba(255,255,255,0.06)"
                  tick={{ fill: '#48484A', fontSize: 11 }}
                  label={{ value: 'Market Price', position: 'insideBottom', offset: -5, fill: '#48484A', fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  stroke="rgba(255,255,255,0.06)"
                  tick={{ fill: '#48484A', fontSize: 11 }}
                  label={{ value: 'Calibrated Prob', angle: -90, position: 'insideLeft', offset: 10, fill: '#48484A', fontSize: 11 }}
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
                  formatter={(value: number | string | undefined, name: string | undefined) => {
                    const v = typeof value === 'number' ? value : Number(value) || 0
                    return [
                      `${(v * 100).toFixed(1)}%`,
                      name === 'calibrated_price' ? 'Calibrated' : name === 'diagonal' ? 'Perfect' : 'Market',
                    ]
                  }}
                />
                {/* Perfect calibration diagonal line */}
                <Line
                  dataKey="diagonal"
                  stroke="rgba(255,255,255,0.15)"
                  strokeDasharray="6 4"
                  dot={false}
                  name="Perfect calibration"
                  legendType="none"
                />
                {/* Calibrated points */}
                <Scatter
                  dataKey="calibrated_price"
                  fill="#C4A24D"
                  fillOpacity={0.9}
                  r={5}
                  name="calibrated_price"
                  line={{ stroke: '#C4A24D', strokeWidth: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-4 mt-3 text-[11px]" style={{ color: 'var(--text-3)' }}>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5" style={{ background: 'rgba(255,255,255,0.15)', borderTop: '1px dashed rgba(255,255,255,0.15)' }} />
              Perfect calibration
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: '#C4A24D' }} />
              Isotonic model
            </span>
            <span>
              Points above diagonal = market overpriced (buy NO), below = underpriced (buy YES)
            </span>
          </div>
        </div>
      )}

      {/* Accuracy Metrics (if trained) */}
      {accuracy?.trained && accuracy.metrics && (
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <BarChart3 className="h-5 w-5" style={{ color: 'var(--accent)' }} />
            <div>
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                Model Accuracy
              </p>
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                Brier score comparison (lower is better) &middot; {accuracy.metrics.n_train} train / {accuracy.metrics.n_test} test samples
              </p>
            </div>
            {brierImprovement !== null && (
              <span className="pill pill-accent ml-auto">
                {brierImprovement > 0 ? '+' : ''}{brierImprovement.toFixed(0)}% vs baseline
              </span>
            )}
          </div>

          {/* Brier Score Bars */}
          <div className="space-y-3">
            {[
              { name: 'Market Baseline', score: accuracy.metrics.baseline_brier, color: 'var(--text-3)' },
              ...(accuracy.metrics.oof_brier ? [
                { name: 'Calibration (OOF)', score: accuracy.metrics.oof_brier.calibration, color: 'var(--blue)' },
                { name: 'XGBoost (OOF)', score: accuracy.metrics.oof_brier.xgboost, color: '#F59E0B' },
                { name: 'LightGBM (OOF)', score: accuracy.metrics.oof_brier.lightgbm, color: '#8B5CF6' },
              ] : []),
              { name: 'Calibration (Test)', score: accuracy.metrics.calibration_brier, color: 'var(--blue)' },
              { name: 'Ensemble (Test)', score: accuracy.metrics.ensemble_brier, color: 'var(--accent)' },
            ].filter(m => m.score !== undefined).map((model) => {
              const maxBrier = accuracy.metrics.baseline_brier || 0.25
              const pct = Math.min(100, (model.score / maxBrier) * 100)
              return (
                <div key={model.name} className="flex items-center gap-3">
                  <p className="text-[11px] w-36 text-right flex-shrink-0" style={{ color: 'var(--text-2)' }}>
                    {model.name}
                  </p>
                  <div className="flex-1 h-5 rounded-lg overflow-hidden" style={{ background: 'rgba(255,255,255,0.03)' }}>
                    <div
                      className="h-full rounded-lg transition-all duration-500"
                      style={{ width: `${pct}%`, background: model.color, opacity: 0.8 }}
                    />
                  </div>
                  <p className="text-[11px] font-mono w-14 text-right flex-shrink-0" style={{ color: model.color }}>
                    {model.score.toFixed(4)}
                  </p>
                </div>
              )
            })}
          </div>

          {/* Weight Breakdown */}
          {accuracy.weights && (
            <div className="mt-5 pt-5" style={{ borderTop: '1px solid var(--border)' }}>
              <p className="text-[11px] uppercase mb-3" style={{ color: 'var(--text-3)' }}>Ensemble Weights</p>
              <div className="flex gap-3 h-3 rounded-full overflow-hidden">
                {Object.entries(accuracy.weights).map(([name, weight]) => (
                  <div
                    key={name}
                    className="h-full rounded-full"
                    style={{
                      width: `${(weight as number) * 100}%`,
                      background: name === 'calibration' ? 'var(--blue)' : name === 'xgboost' ? '#F59E0B' : '#8B5CF6',
                      minWidth: '8px',
                    }}
                    title={`${name}: ${((weight as number) * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <div className="flex gap-4 mt-2">
                {Object.entries(accuracy.weights).map(([name, weight]) => (
                  <span key={name} className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                    {name}: {((weight as number) * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Feature Importance */}
          {accuracy.metrics.xgb_feature_importance && (
            <div className="mt-5 pt-5" style={{ borderTop: '1px solid var(--border)' }}>
              <p className="text-[11px] uppercase mb-3" style={{ color: 'var(--text-3)' }}>Top Features (XGBoost)</p>
              <div className="space-y-2">
                {Object.entries(accuracy.metrics.xgb_feature_importance)
                  .filter(([, v]) => (v as number) > 0)
                  .slice(0, 5)
                  .map(([name, importance]) => {
                    const pct = (importance as number) * 100
                    return (
                      <div key={name} className="flex items-center gap-3">
                        <p className="text-[11px] w-40 text-right font-mono flex-shrink-0" style={{ color: 'var(--text-2)' }}>
                          {name}
                        </p>
                        <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.03)' }}>
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${pct}%`, background: '#F59E0B', opacity: 0.7 }}
                          />
                        </div>
                        <p className="text-[10px] font-mono w-10 text-right" style={{ color: 'var(--text-3)' }}>
                          {pct.toFixed(0)}%
                        </p>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Methodology + Profit Simulation */}
      {accuracy?.trained && accuracy.metrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Training Methodology */}
          <div className="card p-6">
            <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
              Training Methodology
            </p>
            <div className="space-y-3">
              {[
                { label: 'Temporal Split', value: accuracy.metrics?.temporal_split_date || 'Yes', icon: '⏱' },
                { label: 'Walk-Forward CV', value: 'Stratified K-Fold', icon: '🔄' },
                { label: 'Significance Gating', value: 'Wilcoxon signed-rank', icon: '📊' },
                { label: 'Total Resolved', value: `${(accuracy.metrics.n_total_resolved ?? 0).toLocaleString()} markets`, icon: '📈' },
                { label: 'Usable (volume>0)', value: `${(accuracy.metrics.n_usable ?? 0).toLocaleString()} (${((accuracy.metrics.n_usable ?? 0) / Math.max(accuracy.metrics.n_total_resolved ?? 1, 1) * 100).toFixed(1)}%)`, icon: '✅' },
                { label: 'Train / Test', value: `${accuracy.metrics.n_train ?? 0} / ${accuracy.metrics.n_test ?? 0}`, icon: '📋' },
                { label: 'Class Balance', value: `${(accuracy.metrics.class_balance_yes_pct ?? 0).toFixed(1)}% YES`, icon: '⚖' },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between py-1.5" style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <span className="text-[12px]" style={{ color: 'var(--text-2)' }}>
                    {item.icon} {item.label}
                  </span>
                  <span className="text-[12px] font-mono font-medium" style={{ color: 'var(--text)' }}>
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Profit Simulation */}
          {accuracy.profit_simulation && (
            <div className="card p-6">
              <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
                Profit Simulation (Test Set)
              </p>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {[
                  { label: 'Total P&L', value: `$${(accuracy.profit_simulation as any).gated_pnl?.toFixed(1) ?? '0'}`, color: 'var(--green)' },
                  { label: 'Win Rate', value: `${(((accuracy.profit_simulation as any).gated_win_rate ?? 0) * 100).toFixed(1)}%`, color: 'var(--accent)' },
                  { label: 'Trades', value: `${(accuracy.profit_simulation as any).gated_trades ?? 0}`, color: 'var(--text)' },
                  { label: 'Min Edge', value: `${((accuracy.profit_simulation as any).min_net_edge ?? 0.03) * 100}%`, color: 'var(--text-2)' },
                ].map((s) => (
                  <div key={s.label} className="text-center py-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)' }}>
                    <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
                    <p className="text-[18px] font-bold" style={{ color: s.color }}>{s.value}</p>
                  </div>
                ))}
              </div>
              <div className="pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                  Simulated using Kelly-sized paper trades on the held-out test set.
                  Quality-gated: only markets passing volume, liquidity, and price range filters.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Ablation Study */}
      {accuracy?.trained && (accuracy as any).ablation && (
        <div className="card p-6">
          <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
            Ablation Study (Brier Score)
          </p>
          <p className="text-[11px] mb-3" style={{ color: 'var(--text-3)' }}>
            Does every model component help? Lower is better.
          </p>
          <div className="space-y-2">
            {[
              { name: 'Calibration Only', score: (accuracy as any).ablation.calibration_only, color: 'var(--blue)' },
              { name: 'Cal + XGBoost', score: (accuracy as any).ablation.cal_plus_xgb, color: '#F59E0B' },
              { name: 'Cal + LightGBM', score: (accuracy as any).ablation.cal_plus_lgb, color: '#8B5CF6' },
              { name: 'Full Ensemble', score: (accuracy as any).ablation.full_ensemble, color: 'var(--accent)' },
            ].filter(m => m.score != null).map((model) => {
              const maxBrier = (accuracy as any).ablation.calibration_only || 0.1
              const pct = Math.min(100, (model.score / maxBrier) * 100)
              return (
                <div key={model.name} className="flex items-center gap-3">
                  <p className="text-[11px] w-32 text-right flex-shrink-0" style={{ color: 'var(--text-2)' }}>
                    {model.name}
                  </p>
                  <div className="flex-1 h-4 rounded-lg overflow-hidden" style={{ background: 'rgba(255,255,255,0.03)' }}>
                    <div
                      className="h-full rounded-lg transition-all duration-500"
                      style={{ width: `${pct}%`, background: model.color, opacity: 0.7 }}
                    />
                  </div>
                  <p className="text-[11px] font-mono w-14 text-right flex-shrink-0" style={{ color: model.color }}>
                    {model.score.toFixed(4)}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Signal Accuracy (Backtest) */}
      <SignalAccuracyChart />

      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex gap-1.5">
          {(['all', 'overpriced', 'underpriced'] as const).map((dir) => (
            <button
              key={dir}
              onClick={() => setDirectionFilter(dir)}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{
                background: directionFilter === dir ? 'var(--card)' : 'transparent',
                color: directionFilter === dir ? 'var(--text)' : 'var(--text-3)',
                border: `1px solid ${directionFilter === dir ? 'var(--border)' : 'transparent'}`,
              }}
            >
              {dir === 'all' ? 'All' : dir.charAt(0).toUpperCase() + dir.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Min Edge:</span>
          <input
            type="number"
            value={minEdge}
            onChange={(e) => setMinEdge(parseFloat(e.target.value) || 0)}
            className="input"
            style={{ width: '80px', padding: '6px 10px', fontSize: '12px' }}
            min={0}
            step={0.5}
          />
          <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>%</span>
        </div>
        <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
          {filteredMarkets.length} results
        </span>
      </div>

      {/* Mispriced Markets */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Top Mispriced Markets
        </p>

        {filteredMarkets.length > 0 ? (
          <div className="space-y-1.5">
            {filteredMarkets.map((m) => {
              const over = m.direction === 'overpriced'
              const confidence = Math.abs(m.delta_pct)
              const confLabel = confidence > 10 ? 'High' : confidence > 5 ? 'Med' : 'Low'
              return (
                <div
                  key={m.market_id}
                  onClick={() => navigate(`/markets/${m.market_id}`)}
                  className="card card-hover flex items-center gap-4 px-5 py-4 cursor-pointer group"
                >
                  {/* Direction indicator */}
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{
                      background: over ? 'rgba(207,102,121,0.1)' : 'rgba(76,175,112,0.1)',
                    }}
                  >
                    {over ? (
                      <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
                    ) : (
                      <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
                    )}
                  </div>

                  {/* Question */}
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-medium line-clamp-1" style={{ color: 'var(--text)' }}>
                      {m.question}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {m.category && <span className="pill">{m.category}</span>}
                      <span className={`pill ${over ? 'pill-red' : 'pill-green'}`}>
                        {over ? 'Overpriced' : 'Underpriced'}
                      </span>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-5 flex-shrink-0">
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                      <p className="text-[12px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(m.price_yes * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Calibrated</p>
                      <p className="text-[12px] font-mono font-medium" style={{ color: 'var(--blue)' }}>
                        {(m.calibrated_price * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Edge</p>
                      <p className="text-[12px] font-mono font-medium" style={{ color: 'var(--accent)' }}>
                        {(m.edge_estimate * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Confidence</p>
                      <div className="flex items-center gap-1.5">
                        <div className="w-10 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, confidence * 10)}%`,
                              background: 'var(--accent)',
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>
                          {confLabel}
                        </span>
                      </div>
                    </div>
                  </div>

                  <ChevronRight
                    className="h-4 w-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: 'var(--text-3)' }}
                  />
                </div>
              )
            })}
          </div>
        ) : (
          <div className="card flex flex-col items-center py-16">
            <Brain className="h-6 w-6 mb-3" style={{ color: 'var(--text-3)' }} />
            <p className="text-[13px]" style={{ color: 'var(--text-2)' }}>No mispriced markets detected</p>
            <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>Try adjusting the filters above</p>
          </div>
        )}
      </div>
    </div>
  )
}
