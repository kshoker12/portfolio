/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Include all React component files
  ],
  theme: {
    extend: {
      colors: {
        primary: "#2c2e45",
        alt:'#6c71a6',
        primarydark: "#1a1b2a",
        success: "#4f6962",
      }
    },
  },
  plugins: [],
};

