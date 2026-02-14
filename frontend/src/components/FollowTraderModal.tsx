import { useState } from 'react'
import { X, UserPlus, Bell, Zap, DollarSign, AlertTriangle } from 'lucide-react'

interface FollowTraderModalProps {
  traderName: string
  traderId: string
  onClose: () => void
  onConfirm: (settings: FollowSettings) => void
}

export interface FollowSettings {
  trader_id: string
  allocation_amount: number
  copy_percentage: number
  max_position_size: number | null
  auto_copy: boolean
}

export default function FollowTraderModal({
  traderName,
  traderId,
  onClose,
  onConfirm,
}: FollowTraderModalProps) {
  const [mode, setMode] = useState<'manual' | 'auto'>('manual')
  const [allocation, setAllocation] = useState(1000)
  const [copyPercentage, setCopyPercentage] = useState(100)
  const [maxPositionSize, setMaxPositionSize] = useState<number>(500)

  const handleSubmit = () => {
    const settings: FollowSettings = {
      trader_id: traderId,
      allocation_amount: allocation,
      copy_percentage: copyPercentage / 100,
      max_position_size: mode === 'auto' ? maxPositionSize : null,
      auto_copy: mode === 'auto',
    }
    onConfirm(settings)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.8)' }}
      onClick={onClose}
    >
      <div
        className="card w-full max-w-md p-6"
        onClick={(e) => e.stopPropagation()}
        style={{ maxHeight: '90vh', overflowY: 'auto' }}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-[18px] font-bold" style={{ color: 'var(--text)' }}>
              Follow {traderName}
            </h2>
            <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>
              Configure copy trading settings
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
            style={{ color: 'var(--text-3)' }}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Mode Selection */}
        <div className="mb-6">
          <p className="text-[11px] uppercase mb-3 font-semibold" style={{ color: 'var(--text-3)' }}>
            Copy Trading Mode
          </p>
          <div className="grid grid-cols-2 gap-3">
            {/* Manual Mode */}
            <button
              onClick={() => setMode('manual')}
              className="p-4 rounded-xl text-left transition-all"
              style={{
                background: mode === 'manual' ? 'var(--card)' : 'rgba(255,255,255,0.03)',
                border: `2px solid ${mode === 'manual' ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{
                    background: mode === 'manual' ? 'var(--accent-dim)' : 'rgba(255,255,255,0.05)',
                  }}
                >
                  <Bell
                    className="h-4 w-4"
                    style={{ color: mode === 'manual' ? 'var(--accent)' : 'var(--text-3)' }}
                  />
                </div>
                <p
                  className="text-[13px] font-semibold"
                  style={{ color: mode === 'manual' ? 'var(--text)' : 'var(--text-2)' }}
                >
                  Manual
                </p>
              </div>
              <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                See trades, get notifications. You manually decide whether to copy.
              </p>
            </button>

            {/* Auto Mode */}
            <button
              onClick={() => setMode('auto')}
              className="p-4 rounded-xl text-left transition-all"
              style={{
                background: mode === 'auto' ? 'var(--card)' : 'rgba(255,255,255,0.03)',
                border: `2px solid ${mode === 'auto' ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{
                    background: mode === 'auto' ? 'var(--accent-dim)' : 'rgba(255,255,255,0.05)',
                  }}
                >
                  <Zap
                    className="h-4 w-4"
                    style={{ color: mode === 'auto' ? 'var(--accent)' : 'var(--text-3)' }}
                  />
                </div>
                <p
                  className="text-[13px] font-semibold"
                  style={{ color: mode === 'auto' ? 'var(--text)' : 'var(--text-2)' }}
                >
                  Auto
                </p>
              </div>
              <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                Automatically copy all trades instantly. Hands-free trading.
              </p>
            </button>
          </div>
        </div>

        {/* Settings */}
        <div className="space-y-4 mb-6">
          {/* Allocation Amount */}
          <div>
            <label className="text-[12px] font-medium mb-2 flex items-center gap-2" style={{ color: 'var(--text-2)' }}>
              <DollarSign className="h-3.5 w-3.5" />
              Allocation Amount
            </label>
            <input
              type="number"
              value={allocation}
              onChange={(e) => setAllocation(Number(e.target.value))}
              className="w-full px-4 py-2.5 rounded-lg text-[13px] font-mono"
              style={{
                background: 'var(--card)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
              min="100"
              step="100"
            />
            <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>
              Total capital allocated for copying this trader
            </p>
          </div>

          {/* Auto Mode Settings */}
          {mode === 'auto' && (
            <>
              {/* Copy Percentage */}
              <div>
                <label className="text-[12px] font-medium mb-2 flex items-center justify-between" style={{ color: 'var(--text-2)' }}>
                  <span>Copy Percentage</span>
                  <span className="font-mono" style={{ color: 'var(--accent)' }}>
                    {copyPercentage}%
                  </span>
                </label>
                <input
                  type="range"
                  value={copyPercentage}
                  onChange={(e) => setCopyPercentage(Number(e.target.value))}
                  className="w-full"
                  min="10"
                  max="100"
                  step="10"
                  style={{
                    accentColor: 'var(--accent)',
                  }}
                />
                <div className="flex justify-between text-[10px] mt-1" style={{ color: 'var(--text-3)' }}>
                  <span>10%</span>
                  <span>100%</span>
                </div>
                <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>
                  Size of copied positions relative to trader's position size
                </p>
              </div>

              {/* Max Position Size */}
              <div>
                <label className="text-[12px] font-medium mb-2 flex items-center gap-2" style={{ color: 'var(--text-2)' }}>
                  <AlertTriangle className="h-3.5 w-3.5" style={{ color: 'var(--accent)' }} />
                  Max Position Size
                </label>
                <input
                  type="number"
                  value={maxPositionSize}
                  onChange={(e) => setMaxPositionSize(Number(e.target.value))}
                  className="w-full px-4 py-2.5 rounded-lg text-[13px] font-mono"
                  style={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    color: 'var(--text)',
                  }}
                  min="50"
                  step="50"
                />
                <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>
                  Maximum size for any single copied position (risk limit)
                </p>
              </div>

              {/* Warning Banner */}
              <div
                className="p-3 rounded-lg"
                style={{ background: 'rgba(196,162,77,0.1)', border: '1px solid rgba(196,162,77,0.3)' }}
              >
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 mt-0.5" style={{ color: 'var(--accent)' }} />
                  <div>
                    <p className="text-[12px] font-semibold mb-1" style={{ color: 'var(--accent)' }}>
                      Auto-Copy Active
                    </p>
                    <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                      Trades will be executed automatically. Monitor your positions regularly.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2.5 rounded-lg text-[13px] font-semibold transition-colors"
            style={{
              background: 'transparent',
              border: '1px solid var(--border)',
              color: 'var(--text-2)',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="flex-1 px-4 py-2.5 rounded-lg text-[13px] font-semibold transition-colors flex items-center justify-center gap-2"
            style={{
              background: 'var(--accent)',
              color: '#000',
            }}
          >
            <UserPlus className="h-4 w-4" />
            Follow Trader
          </button>
        </div>

        {/* Info Footer */}
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-[10px] text-center" style={{ color: 'var(--text-3)' }}>
            You can modify these settings or unfollow at any time
          </p>
        </div>
      </div>
    </div>
  )
}
