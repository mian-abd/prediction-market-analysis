# Frontend Deployment Guide

Backend is already deployed: `https://prediction-market-analysis-production-e6fa.up.railway.app`

## Quick Deploy Options

### Option 1: Vercel (Recommended) ⭐

**Pros:** Fast, free tier, excellent DX, auto-preview deployments
**Best for:** React/Vite apps

#### Steps:
1. Install Vercel CLI (optional):
   ```bash
   npm install -g vercel
   ```

2. **Via GitHub (Recommended):**
   - Push your code to GitHub
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your repository
   - Set Root Directory to `frontend`
   - Add environment variable:
     - `VITE_API_URL` = `https://prediction-market-analysis-production-e6fa.up.railway.app`
   - Deploy!

3. **Via CLI:**
   ```bash
   cd frontend
   vercel
   # Follow prompts, set root to current directory
   # Add env var: VITE_API_URL=https://prediction-market-analysis-production-e6fa.up.railway.app
   ```

4. **Set Production URL:**
   ```bash
   vercel --prod
   ```

### Option 2: Netlify

**Pros:** Great free tier, easy deploys, built-in forms/functions
**Best for:** Static sites, JAMstack

#### Steps:
1. **Via GitHub:**
   - Push code to GitHub
   - Go to [netlify.com](https://netlify.com)
   - Click "Add new site" → "Import existing project"
   - Connect to GitHub, select repo
   - Set Base directory to `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/dist`
   - Add environment variable:
     - `VITE_API_URL` = `https://prediction-market-analysis-production-e6fa.up.railway.app`
   - Deploy!

2. **Via CLI:**
   ```bash
   npm install -g netlify-cli
   cd frontend
   netlify deploy --prod
   ```

### Option 3: Cloudflare Pages

**Pros:** Fastest CDN, generous free tier, workers integration
**Best for:** Global performance

#### Steps:
1. Go to [pages.cloudflare.com](https://pages.cloudflare.com)
2. Connect your GitHub repo
3. Set build configuration:
   - Framework preset: Vite
   - Build command: `npm run build`
   - Build output directory: `dist`
   - Root directory: `frontend`
4. Add environment variable:
   - `VITE_API_URL` = `https://prediction-market-analysis-production-e6fa.up.railway.app`
5. Deploy!

### Option 4: Railway (Same Platform as Backend)

**Pros:** Everything in one place, simple deployment
**Cons:** More expensive for frontend (static sites)

#### Steps:
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Click "Add Service" → set Root Directory to `frontend`
5. Add environment variable:
   - `VITE_API_URL` = `https://prediction-market-analysis-production-e6fa.up.railway.app`
6. Set build command: `npm run build`
7. Set start command: `npm run preview` (or use a static server)
8. Deploy!

## Testing Build Locally

Before deploying, test the production build:

```bash
cd frontend

# Build for production
npm run build

# Preview the build locally
npm run preview
# Opens at http://localhost:4173
```

Visit the preview URL and verify:
- ✅ API calls work (check Network tab)
- ✅ All pages load correctly
- ✅ No console errors
- ✅ Charts render properly

## CORS Configuration

Your Railway backend needs to allow requests from your frontend domain.

Check `backend/main.py` has this CORS config:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev
        "http://localhost:4173",  # Vite preview
        "https://your-frontend-domain.vercel.app",  # Add your prod domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**After deploying frontend, update the backend CORS config with your new domain!**

## Post-Deployment Checklist

- [ ] Frontend deployed successfully
- [ ] Environment variable `VITE_API_URL` is set
- [ ] Backend CORS updated with frontend domain
- [ ] Test all major features:
  - [ ] Dashboard loads
  - [ ] Market browser works
  - [ ] Charts render
  - [ ] API calls succeed (check DevTools Network tab)
  - [ ] Auto-trading panel shows data
- [ ] No console errors
- [ ] Custom domain configured (optional)

## Troubleshooting

### API calls failing (404/CORS errors)
- Check `VITE_API_URL` is set correctly (no trailing slash)
- Check backend CORS allows your frontend domain
- Check DevTools Network tab for actual request URL

### Blank page / "Failed to load module"
- Clear browser cache
- Rebuild: `npm run build`
- Check build output in `dist/` folder

### Environment variables not working
- Vite env vars MUST start with `VITE_`
- Rebuild after changing `.env` files
- Check with: `console.log(import.meta.env.VITE_API_URL)`

## Recommended: Vercel
For this project, I recommend **Vercel** because:
- ✅ Best React/Vite support
- ✅ Free tier is generous
- ✅ Automatic HTTPS
- ✅ GitHub integration for auto-deploys
- ✅ Preview deployments for PRs
- ✅ Fast global CDN
