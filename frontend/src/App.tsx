import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  ArrowLeftRight,
  Store,
  Brain,
  Target,
  Activity,
  TrendingUp,
  Briefcase,
  Users,
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import ArbitrageScanner from './pages/ArbitrageScanner'
import MarketBrowser from './pages/MarketBrowser'
import MarketDetail from './pages/MarketDetail'
import MLModels from './pages/MLModels'
import CalibrationChart from './pages/CalibrationChart'
import Analytics from './pages/Analytics'
import Portfolio from './pages/Portfolio'
import TraderLeaderboard from './pages/TraderLeaderboard'
import TraderDetail from './pages/TraderDetail'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/markets', icon: Store, label: 'Markets' },
  { to: '/arbitrage', icon: ArrowLeftRight, label: 'Arbitrage' },
  { to: '/portfolio', icon: Briefcase, label: 'Portfolio' },
  { to: '/copy-trading', icon: Users, label: 'Copy Trading' },
  { to: '/models', icon: Brain, label: 'ML Models' },
  { to: '/calibration', icon: Target, label: 'Calibration' },
  { to: '/analytics', icon: TrendingUp, label: 'Analytics' },
]

function App() {
  const location = useLocation()

  return (
    <div className="flex h-screen" style={{ background: 'var(--bg)' }}>
      {/* Sidebar */}
      <aside
        className="w-[240px] flex-shrink-0 flex flex-col"
        style={{ borderRight: '1px solid var(--border)' }}
      >
        {/* Logo */}
        <div className="px-6 pt-7 pb-8">
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: 'var(--accent-dim)' }}
            >
              <Activity className="h-[17px] w-[17px]" style={{ color: 'var(--accent)' }} />
            </div>
            <div>
              <p className="text-[15px] font-semibold" style={{ color: 'var(--text)' }}>
                PredictFlow
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 space-y-0.5">
          {navItems.map(({ to, icon: Icon, label }) => {
            const isActive =
              to === '/' ? location.pathname === '/' : location.pathname.startsWith(to)
            return (
              <NavLink
                key={to}
                to={to}
                className="flex items-center gap-3 px-4 py-2.5 rounded-xl text-[13px] font-medium transition-colors duration-150"
                style={{
                  color: isActive ? 'var(--text)' : 'var(--text-3)',
                  background: isActive ? 'var(--card)' : 'transparent',
                }}
              >
                <Icon className="h-[17px] w-[17px]" />
                {label}
              </NavLink>
            )
          })}
        </nav>

        {/* Status */}
        <div className="px-6 py-5" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2">
            <span
              className="h-1.5 w-1.5 rounded-full pulse-dot"
              style={{ background: 'var(--green)' }}
            />
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Live
            </span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <div className="px-10 py-10 max-w-[1100px] mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/arbitrage" element={<ArbitrageScanner />} />
            <Route path="/markets" element={<MarketBrowser />} />
            <Route path="/markets/:id" element={<MarketDetail />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/copy-trading" element={<TraderLeaderboard />} />
            <Route path="/copy-trading/:traderId" element={<TraderDetail />} />
            <Route path="/models" element={<MLModels />} />
            <Route path="/calibration" element={<CalibrationChart />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
