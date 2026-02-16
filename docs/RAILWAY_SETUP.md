# Railway Backend Configuration Checklist

## ✅ What I Fixed

### 1. PostgreSQL Database Support
- ✅ Added `asyncpg` driver to requirements.txt
- ✅ Auto-converts Railway's `DATABASE_URL` to async format
- ✅ Made SQLite-specific config conditional
- ✅ **Result:** Your portfolio will now persist across deployments!

### 2. CORS Configuration
- ✅ Backend uses `FRONTEND_URL` environment variable
- ✅ Allows requests from Vercel frontend

### 3. Deployment Config
- ✅ Procfile configured for Railway
- ✅ Nixpacks.toml for system dependencies

---

## 🔧 Railway Configuration Required

### Current Variables (I can see you have):
```
✅ DATABASE_URL=${{Postgres.DATABASE_URL}}
✅ FRONTEND_URL=https://prediction-market-analysis-one.vercel.app
✅ RAILPACK_DEPLOY_APT_PACKAGES=libgomp1
```

### Missing Variable (Add this):
```
⚠️  ANTHROPIC_API_KEY=sk-ant-...
```
**Without this, Claude AI features won't work** (market analysis, etc.)

---

## 📋 Deployment Steps

### 1. Add Missing Environment Variable
Go to Railway → Variables → Add New Variable:
```
Name:  ANTHROPIC_API_KEY
Value: sk-ant-... (your actual API key)
```

### 2. Trigger Redeploy
Railway should automatically redeploy when you push. If not:
- Go to Railway → Deployments
- Click "Redeploy" on the latest deployment

### 3. Wait for Deployment (~2-3 minutes)
Check the logs to ensure:
- ✅ Build succeeds
- ✅ Database connection established
- ✅ No errors in startup

### 4. Verify Backend Health
Visit: `https://prediction-market-analysis-production-e6fa.up.railway.app/api/v1/system/health`

Should return:
```json
{"status": "healthy"}
```

### 5. Test Frontend
Visit: `https://prediction-market-analysis-one.vercel.app`

Check:
- ✅ No CORS errors (F12 → Console)
- ✅ Dashboard loads with data
- ✅ API calls succeed (F12 → Network tab)
- ✅ Charts render

---

## 🗄️ About the Database Switch

### Before (SQLite):
- ❌ Data stored in container filesystem
- ❌ Lost on every redeploy
- ❌ Required volumes (complex setup)

### After (PostgreSQL):
- ✅ Data stored in Railway's managed database
- ✅ Persists across all deployments
- ✅ Better for production use
- ✅ More reliable and scalable

### What happens to old data?
- Your previous deployments used ephemeral SQLite
- That data is already gone from past redeploys
- Fresh start with PostgreSQL is actually better!
- New data will persist from now on

---

## 🧪 How to Verify Persistence

1. **Make a paper trade or let auto-trading run**
2. **Trigger a redeploy** (push a small change or click "Redeploy")
3. **Check if data persists**:
   - Visit: `/api/v1/auto-trading/status`
   - Your positions should still be there! ✅

---

## 🚨 Common Issues & Fixes

### Issue: "CORS policy: No 'Access-Control-Allow-Origin'"
**Fix:**
- Ensure `FRONTEND_URL` is set to your Vercel URL (no trailing slash)
- Redeploy backend after changing the variable

### Issue: "Connection to database failed"
**Fix:**
- Verify Railway PostgreSQL service is running
- Check `DATABASE_URL` variable exists
- Check deployment logs for detailed error

### Issue: "Module 'asyncpg' not found"
**Fix:**
- Make sure latest code is deployed (I just pushed the fix)
- Railway should reinstall dependencies automatically

### Issue: Frontend shows "Network Error"
**Fix:**
- Check backend is actually running (visit `/api/v1/system/health`)
- Check CORS configuration
- Look at browser DevTools → Network tab for actual error

---

## 📊 Environment Variable Reference

| Variable | Value | Required | Purpose |
|----------|-------|----------|---------|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | ✅ Yes | PostgreSQL connection |
| `FRONTEND_URL` | `https://prediction-market-analysis-one.vercel.app` | ✅ Yes | CORS whitelist |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | ⚠️  For AI features | Claude API access |
| `RAILPACK_DEPLOY_APT_PACKAGES` | `libgomp1` | ✅ Yes | XGBoost/LightGBM dependency |

---

## 🎯 Next Steps

1. **Add `ANTHROPIC_API_KEY` to Railway** (if you want AI features)
2. **Wait for auto-redeploy** (or trigger manually)
3. **Test the deployed app** at both URLs
4. **Verify data persistence** by making trades and redeploying

Your deployment should now be rock-solid! 🚀

---

## 📝 Notes

- Railway automatically redeploys when you push to `main` branch
- PostgreSQL backups are handled by Railway
- You can view database contents via Railway's PostgreSQL service dashboard
- Local development still uses SQLite (`./data/markets.db`)
