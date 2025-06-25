/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Standard Next.js build for Netlify functions
  images: {
    unoptimized: true
  },
  
  // Environment variables that should be available to the browser
  env: {
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
  }
}

module.exports = nextConfig