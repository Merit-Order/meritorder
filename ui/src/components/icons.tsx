import type { ReactNode } from "react";

// The console's few icons, drawn inline (16-unit grid, 1.5 strokes in currentColor): no icon font, nothing fetched.
const PATHS = {
  ok: <><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <path d="M5.2 8.2l1.9 1.9 3.8-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></>,
  warn: <><path d="M8 2.2l6.2 11H1.8z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    <path d="M8 6.4v3.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="8" cy="11.5" r=".9" fill="currentColor" /></>,
  bad: <><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <path d="M5.8 5.8l4.4 4.4M10.2 5.8l-4.4 4.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></>,
  info: <><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <path d="M8 7.2v4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="8" cy="4.9" r=".9" fill="currentColor" /></>,
  off: <><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <path d="M5.2 8h5.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></>,
  clock: <><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <path d="M8 4.6V8l2.3 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></>,
  pin: <path d="M9.8 1.8l4.4 4.4-2 .6-2.6 2.6.2 3-1.4 1.4-2.6-2.6-3.3 3.3-.8-.8 3.3-3.3-2.6-2.6 1.4-1.4 3 .2 2.6-2.6z" fill="none"
    stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />,
  stop: <rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />,
  auto: <path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.8v2.6h-2.6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />,
  chev: <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  down: <path d="M8 2.5v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />,
  arrow: <path d="M3 8h9M8.5 4.5L12 8l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  drain: <path d="M3 4h10M4.5 8h7M6.5 12h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />,
  theme: <><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M8 1.5a6.5 6.5 0 0 1 0 13z" fill="currentColor" /></>,
  menu: <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />,
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false">{PATHS[name]}</svg>;
}

/** The brand mark: three rising bars in a rounded square (the merit order, cheapest first). */
export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden="true" focusable="false">
      <rect x="1" y="1" width="20" height="20" rx="5" fill="var(--mantine-color-text)" />
      <path d="M6 15.5h2.5v-3H6zM9.75 15.5h2.5v-6h-2.5zM13.5 15.5H16v-9h-2.5z" fill="var(--mo-surface)" />
    </svg>
  );
}
