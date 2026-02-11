import { Routes, Route, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  ArrowLeftRight,
  Store,
  Brain,
  Target,
  TrendingUp,
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import ArbitrageScanner from './pages/ArbitrageScanner'
import MarketBrowser from './pages/MarketBrowser'
import MarketDetail from './pages/MarketDetail'
import MLModels from './pages/MLModels'
import CalibrationChart from './pages/CalibrationChart'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/markets', icon: Store, label: 'Markets' },
  { to: '/arbitrage', icon: ArrowLeftRight, label: 'Arbitrage' },
  { to: '/models', icon: Brain, label: 'ML Models' },
  { to: '/calibration', icon: Target, label: 'Calibration' },
]

function App() {
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-gray-950 border-r border-gray-800 flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-5 border-b border-gray-800">
          <TrendingUp className="h-7 w-7 text-blue-500" />
          <div>
            <h1 className="text-sm font-bold tracking-wide text-white">
              PredictFlow
            </h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest">
              Market Analysis
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse-dot" />
            Live data feed
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/arbitrage" element={<ArbitrageScanner />} />
            <Route path="/markets" element={<MarketBrowser />} />
            <Route path="/markets/:id" element={<MarketDetail />} />
            <Route path="/models" element={<MLModels />} />
            <Route path="/calibration" element={<CalibrationChart />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
