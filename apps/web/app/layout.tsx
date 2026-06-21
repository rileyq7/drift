import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Drift — agents as declarative blocks",
  description:
    "An intent-based language for LLM agents. Write agents in English. Transpile to async Python.",
  openGraph: {
    title: "Drift — agents as declarative blocks",
    description:
      "An intent-based language for LLM agents. Write agents in English. Transpile to async Python.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Drift — agents as declarative blocks",
    description:
      "An intent-based language for LLM agents. Write agents in English. Transpile to async Python.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
