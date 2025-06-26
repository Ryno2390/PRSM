/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // PRSM UI Mockup Color Scheme
        'prsm': {
          // Dark theme colors
          'bg-primary': '#000000',      // Black
          'bg-secondary': '#1a1a1a',    // Dark Grey
          'bg-tertiary': '#333333',     // Medium Grey
          'text-primary': '#ffffff',    // White
          'text-secondary': '#b3b3b3',  // Light Grey
          'border': '#4d4d4d',          // Grey border
          'error': '#f85149',           // Red for errors
          
          // Light theme colors
          'light-bg-primary': '#ffffff',    // White
          'light-bg-secondary': '#f0f0f0',  // Very Light Grey
          'light-bg-tertiary': '#d9d9d9',   // Light Grey
          'light-text-primary': '#000000',  // Black
          'light-text-secondary': '#4d4d4d', // Dark Grey
          'light-border': '#cccccc',        // Grey border
          'light-error': '#c0392b',         // Red for errors
        },
        'prsm-blue': {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        'prsm-indigo': {
          50: '#eef2ff',
          100: '#e0e7ff',
          200: '#c7d2fe',
          300: '#a5b4fc',
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
          800: '#3730a3',
          900: '#312e81',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
  ],
}