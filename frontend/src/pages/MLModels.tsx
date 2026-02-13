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
} from 'lucide-react'
import apiClient from '../api/client'

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
    logistic_brier: number
    calibration_brier: number
    xgboost_brier: number
    lightgbm_brier: number
    ensemble_brier: number
    ensemble_auc: number
    xgb_feature_importance: Record<string, number>
    lgb_feature_importance: Record<string, number>
  }
  weights: Record<string, number>
  models: Record<string, {
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
    ? markets.reduce((s, m) => s + m.edge_estimate, 0) / markets.length * 100
    : 0

  const brierImprovement = accuracy?.trained && accuracy.metrics
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
              { label: 'Brier Score', value: accuracy?.models?.calibration?.brier_score?.toFixed(4) ?? '+6pp bias', color: 'var(--accent)' },
              { label: 'Weight', value: accuracy?.weights?.calibration ? `${(accuracy.weights.calibration * 100).toFixed(0)}%` : '100%', color: 'var(--blue)' },
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
          {accuracy?.trained ? (
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Ensemble Brier', value: accuracy.metrics.ensemble_brier.toFixed(4), color: 'var(--accent)' },
                { label: 'AUC-ROC', value: accuracy.metrics.ensemble_auc.toFixed(3), color: 'var(--green)' },
                { label: 'Training Set', value: `${accuracy.metrics.n_usable}`, color: 'var(--text)' },
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
              { name: 'Logistic Regression', score: accuracy.metrics.logistic_brier, color: 'var(--text-2)' },
              { name: 'Calibration (Isotonic)', score: accuracy.metrics.calibration_brier, color: 'var(--blue)' },
              { name: 'XGBoost', score: accuracy.metrics.xgboost_brier, color: '#F59E0B' },
              { name: 'LightGBM', score: accuracy.metrics.lightgbm_brier, color: '#8B5CF6' },
              { name: 'Ensemble', score: accuracy.metrics.ensemble_brier, color: 'var(--accent)' },
            ].map((model) => {
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
