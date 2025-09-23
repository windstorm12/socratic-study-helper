# Deployment Guide

## Environment Variables Required

Set these environment variables in your Railway deployment:

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `PORT`: Port number (Railway will set this automatically)

## Railway Backend Deployment

1. Connect your GitHub repository to Railway
2. Set the environment variable `GEMINI_API_KEY` in Railway dashboard
3. Railway will automatically detect the Python app and deploy it

## Vercel Frontend Deployment

The frontend is already configured to work with Railway backend. The `BACKEND_URL` in `templates/index.html` points to your Railway deployment.

For Vercel deployment:
1. Create a new Vercel project
2. Connect your GitHub repository
3. Set the build command to: `echo "Static site"`
4. Set the output directory to: `templates`
5. Deploy

## Security Notes

- Never commit your actual API key to the repository
- The API key is now properly handled via environment variables
- CORS is configured to allow cross-origin requests
