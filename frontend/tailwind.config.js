/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      keyframes: {
        bounce_dot: {
          '0%, 80%, 100%': { transform: 'scale(0)' },
          '40%': { transform: 'scale(1.0)' },
        },
      },
      animation: {
        bounce_1: 'bounce_dot 1.4s infinite ease-in-out both',
        bounce_2: 'bounce_dot 1.4s 0.2s infinite ease-in-out both',
        bounce_3: 'bounce_dot 1.4s 0.4s infinite ease-in-out both',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} 