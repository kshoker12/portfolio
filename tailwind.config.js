/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Include all React component files
  ],
  theme: {
    extend: {
      colors: {
        primary: "#2c2e45",
        secondary: "#aaa6c3",
        alt:'#6c71a6',
        primarydark: "#1a1b2a",
        success: "#4f6962",
        darker: "#050816",
        secondarydark: "#aaa6c3",
        tertiary: "#151030",
        "black-100": "#100d25",
        "black-200": "#090325",
        "white-100": "#f3f3f3",
      }
    },
  },
  plugins: [],
};