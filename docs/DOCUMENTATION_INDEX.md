# PredictFlow — Documentation Index

> **Use this as the master index** for all project documentation.  
> **Last updated**: 2026-02-28

---

## Quick links

| Need | Document |
|------|----------|
| **Business view** — what we did, metrics, roadmap, when we're done | [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md) |
| **Technical view** — architecture, APIs, pipeline, ML, execution | [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) |
| **Profitability** — why we lose, research, 8-layer fix plan | [PROFITABILITY_ROADMAP.md](PROFITABILITY_ROADMAP.md) |
| **Audit & implementation** — what was verified, 17 fixes (4 CRITICAL), L4/L6/L8 | [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md) |
| **Production readiness** — audit, 60–90 day plan | [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) |
| **Getting started** | [README.md](../README.md) |
| **What changed** (versions, fixes) | [CHANGELOG.md](../CHANGELOG.md) |

---

## By audience

### Product / business

- [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md) — Vision, current state, P&L, roadmap, “done” criteria  
- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) — Audit results, risks, demo vs production  
- [PROFITABILITY_ROADMAP.md](PROFITABILITY_ROADMAP.md) — Root causes, research, layer-by-layer plan  

### Engineering

- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) — System design, DB, pipeline, API, ML, execution  
- [AUDIT_AND_IMPLEMENTATION.md](AUDIT_AND_IMPLEMENTATION.md) — Deep audit (17 issues, 4 CRITICAL), fixes, L4/L6/L8 implementation log  
- [ARCHITECTURE.md](ARCHITECTURE.md) — Data flow, design decisions (legacy; technical doc is canonical)  
- [API.md](API.md) — Endpoint reference  
- [STRATEGIES.md](STRATEGIES.md) — Trading strategies (theory + implementation)  
- [SETUP.md](SETUP.md) — Local setup  
- [DEPLOYMENT.md](DEPLOYMENT.md), [RAILWAY_SETUP.md](RAILWAY_SETUP.md) — Deployment  
- [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) — Checklist and risks  

### Other

- [AUTOMATION.md](AUTOMATION.md) — Automation notes  
- [WEBSOCKET_STREAMING.md](WEBSOCKET_STREAMING.md), [LOW_LATENCY_ARCHITECTURE.md](LOW_LATENCY_ARCHITECTURE.md) — Streaming and latency  
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md), [UX_IMPROVEMENTS.md](UX_IMPROVEMENTS.md), [ROADMAP_90_DAYS.md](ROADMAP_90_DAYS.md) — Historical context  

---

## Update workflow (after every few hours of work)

1. **Re-run or re-audit** whatever you changed (e.g. pipeline, strategies, API, DB).  
2. **Refresh numbers** in [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md) (Section 3: metrics; Section 6: “done” checklist).  
3. **Refresh** [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) if you added routes, tables, strategies, or config.  
4. **Tell the assistant**: *“Update the documentation for everything we did in this session”* — and point to this index so it knows where to write.

Keeping **BUSINESS** and **TECHNICAL** separate lets you see “are we done?” (business) and “how does it work?” (technical) at a glance.
