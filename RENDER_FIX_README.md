# Fixing Connection Refused on Render

If you are seeing "Connection Refused" or "localhost" errors after deploying to Render, it is because the Frontend is still trying to talk to `localhost:5000` instead of your real Backend URL.

## Step 1: Get your Backend URL
1. Go to your Render Dashboard.
2. Find your **Python/Flask Backend** service.
3. Copy its URL (e.g., `https://ai-detective-backend.onrender.com`).

## Step 2: Configure Frontend
1. Go to your **Static Site (Frontend)** service on Render.
2. Go to **Environment**.
3. Click **Add Environment Variable**.
4. Key: `REACT_APP_API_BASE`
5. Value: `<PASTE_YOUR_BACKEND_URL_HERE>` (Do not add a trailing slash `/`)
   - Example: `https://ai-detective-backend.onrender.com`

## Step 3: Redeploy
1. Click **Manual Deploy** > **Deploy latest commit** (or Clear build cache & deploy).
2. Wait for the build to finish.

The error should now be resolved.
