import { lazy, Suspense, useState, useEffect } from 'react'
import { Routes, Route, NavLink, useLocation, Link } from 'react-router-dom'
import {
  LayoutDashboard,
  Store,
  Zap,
  Activity,
  Briefcase,
  Users,
  Shield,
  Brain,
  Target,
  Network,
  Menu,
  X,
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import MarketBrowser from './pages/MarketBrowser'
import MarketDetail from './pages/MarketDetail'
import Portfolio from './pages/Portfolio'
import TraderLeaderboard from './pages/TraderLeaderboard'
import TraderDetail from './pages/TraderDetail'
import MLModels from './pages/MLModels'
import ArbitrageScanner from './pages/ArbitrageScanner'
import CalibrationChart from './pages/CalibrationChart'
import Analytics from './pages/Analytics'

// Lazy-loaded new pages
const SignalsHub = lazy(() => import('./pages/SignalsHub'))
const SystemHealth = lazy(() => import('./pages/SystemHealth'))

/* ─── Navigation Structure (New IA) ─── */

interface NavSection {
  label?: string
  items: { to: string; icon: typeof LayoutDashboard; label: string }[]
}

const navSections: NavSection[] = [
  {
    items: [
      { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    ],
  },
  {
    label: 'Trading',
    items: [
      { to: '/markets', icon: Store, label: 'Markets' },
      { to: '/signals', icon: Zap, label: 'Signals' },
      { to: '/correlation', icon: Network, label: 'Correlation' },
      { to: '/portfolio', icon: Briefcase, label: 'Portfolio' },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { to: '/models', icon: Brain, label: 'ML Models' },
      { to: '/copy-trading', icon: Users, label: 'Copy Trading' },
      { to: '/calibration', icon: Target, label: 'Calibration' },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/system', icon: Shield, label: 'Data Quality' },
    ],
  },
]

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
        <span className="text-[13px]" style={{ color: 'var(--text-3)' }}>Loading...</span>
      </div>
    </div>
  )
}

function App() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  // Close mobile nav on route change
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  // Close mobile nav on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false)
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [])

  const sidebarContent = (
    <>
      {/* Logo */}
      <Link to="/" className="px-5 pt-6 pb-5 block" style={{ textDecoration: 'none' }}>
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Activity className="h-4 w-4" style={{ color: 'var(--accent)' }} />
          </div>
          <span className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
            PredictFlow
          </span>
        </div>
      </Link>

      {/* Nav Sections */}
      <nav className="flex-1 px-3 space-y-4 overflow-y-auto">
        {navSections.map((section, sIdx) => (
          <div key={sIdx}>
            {section.label && (
              <p
                className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider"
                style={{ color: 'var(--text-3)' }}
              >
                {section.label}
              </p>
            )}
            <div className="space-y-0.5">
              {section.items.map(({ to, icon: Icon, label }) => {
                const isActive =
                  to === '/'
                    ? location.pathname === '/'
                    : location.pathname.startsWith(to)
                return (
                  <NavLink
                    key={to}
                    to={to}
                    className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors duration-150"
                    style={{
                      color: isActive ? 'var(--text)' : 'var(--text-3)',
                      background: isActive ? 'var(--card)' : 'transparent',
                    }}
                  >
                    <Icon className="h-4 w-4 flex-shrink-0" />
                    {label}
                  </NavLink>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Status Footer */}
      <div className="px-5 py-4" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2">
          <span
            className="h-1.5 w-1.5 rounded-full pulse-dot"
            style={{ background: 'var(--green)' }}
          />
          <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
            Pipeline active
          </span>
        </div>
      </div>
    </>
  )

  return (
    <div className="flex h-screen" style={{ background: 'var(--bg)' }}>
      {/* Mobile Header */}
      <div className="mobile-header">
        <button
          onClick={() => setMobileOpen(true)}
          className="p-2 -ml-2 rounded-lg"
          style={{ color: 'var(--text)' }}
          aria-label="Open navigation"
        >
          <Menu className="h-5 w-5" />
        </button>
        <Link to="/" className="flex items-center gap-2" style={{ textDecoration: 'none' }}>
          <div
            className="w-7 h-7 rounded-md flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Activity className="h-3.5 w-3.5" style={{ color: 'var(--accent)' }} />
          </div>
          <span className="text-[13px] font-semibold" style={{ color: 'var(--text)' }}>
            PredictFlow
          </span>
        </Link>
        <div style={{ width: '36px' }} /> {/* Spacer for centering */}
      </div>

      {/* Mobile Overlay */}
      {mobileOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar - Desktop */}
      <aside
        className="sidebar-desktop w-[220px] flex-shrink-0 flex flex-col"
        style={{ borderRight: '1px solid var(--border)' }}
      >
        {sidebarContent}
      </aside>

      {/* Sidebar - Mobile */}
      <aside
        className={`sidebar-mobile flex flex-col ${mobileOpen ? 'sidebar-mobile-open' : ''}`}
      >
        <div className="flex items-center justify-end px-3 pt-3">
          <button
            onClick={() => setMobileOpen(false)}
            className="p-2 rounded-lg"
            style={{ color: 'var(--text-3)' }}
            aria-label="Close navigation"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        {sidebarContent}
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto main-content">
        <div className="px-4 sm:px-8 py-6 sm:py-8 max-w-[1200px] mx-auto">
          <Suspense fallback={<PageLoader />}>
            <Routes>
              {/* Dashboard */}
              <Route path="/" element={<Dashboard />} />

              {/* Markets */}
              <Route path="/markets" element={<MarketBrowser />} />
              <Route path="/markets/:id" element={<MarketDetail />} />

              {/* Signals (new unified hub) */}
              <Route path="/signals" element={<SignalsHub />} />
              <Route path="/signals/ml" element={<MLModels />} />
              <Route path="/signals/arbitrage" element={<ArbitrageScanner />} />
              <Route path="/signals/copy" element={<TraderLeaderboard />} />

              {/* Correlation */}
              <Route path="/correlation" element={<Analytics />} />

              {/* Portfolio */}
              <Route path="/portfolio" element={<Portfolio />} />

              {/* Analysis (existing pages, reorganized) */}
              <Route path="/models" element={<MLModels />} />
              <Route path="/copy-trading" element={<TraderLeaderboard />} />
              <Route path="/copy-trading/:traderId" element={<TraderDetail />} />
              <Route path="/calibration" element={<CalibrationChart />} />

              {/* System (new) */}
              <Route path="/system" element={<SystemHealth />} />

              {/* Legacy routes (redirect support) */}
              <Route path="/arbitrage" element={<ArbitrageScanner />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  )
}

export default App
