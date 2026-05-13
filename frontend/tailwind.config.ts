import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: "#172024",
        mist: "#eef3f1",
        moss: "#536b5b",
        signal: "#2f6f73",
        amberline: "#b7791f",
      },
      boxShadow: {
        soft: "0 18px 60px rgba(18, 27, 30, 0.12)",
      },
    },
  },
  plugins: [],
} satisfies Config;
