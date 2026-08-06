/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#cc785c",
          dark: "#a85f48",
        },
      },
    },
  },
  plugins: [],
};
