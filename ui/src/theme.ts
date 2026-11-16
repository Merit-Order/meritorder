import { createTheme, type CSSVariablesResolver, type MantineColorsTuple } from "@mantine/core";

// "Clean technical": Geist and Geist Mono, one indigo accent (`mo`) for the current place, the primary action and the
// series that matters; hairline-bordered panels; muted greys for every other series; green, amber and red only for
// status, always with an icon and a word. Both schemes are designed; the console opens dark (main.tsx).
const mo: MantineColorsTuple = ["#EEEDFD", "#DEDCFC", "#C9C6F8", "#A9A6FF", "#8C88FF", "#6A64F0", "#4F46E5", "#4338CA", "#3730A3", "#2E2A85"];
// Mantine reads dark[0] text, dark[2] dimmed, dark[4] borders, dark[5] hover, dark[6] default fill, dark[7] body.
const dark: MantineColorsTuple = ["#EDEEF3", "#C3C6D2", "#9A9EAE", "#62677A", "#30333F", "#23252E", "#14151B", "#0B0C0F", "#08090C", "#050608"];
const gray: MantineColorsTuple = ["#F7F7F9", "#F1F1F5", "#E7E7EC", "#D6D6DE", "#C4C6D2", "#9A9EAE", "#62677A", "#3B3E4A", "#24262F", "#0E0F14"];
// Status steps, so color="green" / "yellow" / "red" stay readable in both schemes.
const green: MantineColorsTuple = ["#EAF6EE", "#D3EEDC", "#BCE3C9", "#8FD1A8", "#5CCB8A", "#3BAE6C", "#24924F", "#167A3E", "#0F6131", "#0A4824"];
const yellow: MantineColorsTuple = ["#FFF6E6", "#FDEBC7", "#F0D6A8", "#F2B860", "#E6A33C", "#C98415", "#B06E05", "#9A5B00", "#7C4900", "#5E3700"];
const red: MantineColorsTuple = ["#FDEEEC", "#FAD9D5", "#F3C5BF", "#F58B80", "#EC6A5C", "#DC4636", "#C7321F", "#B42318", "#931C13", "#73150E"];

export const theme = createTheme({
  primaryColor: "mo",
  primaryShade: { light: 6, dark: 4 },
  colors: { mo, dark, gray, green, yellow, red },
  // Self-hosted (@fontsource-variable, SIL OFL 1.1): the console's CSP allows fonts from itself only, and it works offline.
  fontFamily: '"Geist Variable", Geist, system-ui, -apple-system, "Segoe UI", sans-serif',
  fontFamilyMonospace: '"Geist Mono Variable", "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
  headings: { fontFamily: '"Geist Variable", Geist, system-ui, sans-serif', fontWeight: "600" },
  fontSizes: { xs: "12px", sm: "13px", md: "14px", lg: "16px", xl: "20px" },
  radius: { xs: "4px", sm: "6px", md: "8px", lg: "10px", xl: "14px" },
  defaultRadius: "md",
  cursorType: "pointer",
  focusRing: "auto",
  components: {
    Paper: { defaultProps: { withBorder: true, p: "md", radius: "lg", shadow: "none" } },
    Card: { defaultProps: { withBorder: true, radius: "lg" } },
    Badge: { defaultProps: { variant: "light", radius: "sm", size: "md" }, styles: { root: { textTransform: "none", fontWeight: 500, letterSpacing: 0 } } },
    Button: { defaultProps: { radius: "md" }, styles: { root: { fontWeight: 500 } } },
    SegmentedControl: { defaultProps: { radius: "md", size: "xs" } },
    Tabs: { defaultProps: { color: "mo" } },
    Table: { defaultProps: { verticalSpacing: 6, horizontalSpacing: "sm", highlightOnHover: true, striped: false } },
    Progress: { defaultProps: { radius: "xs", size: "sm", color: "mo" } },
    Tooltip: { defaultProps: { withArrow: false, multiline: true, maw: 360 } },
    NavLink: { defaultProps: { color: "mo", variant: "light" } },
    Alert: { defaultProps: { variant: "light", radius: "md" } },
    Modal: { defaultProps: { radius: "lg" } },
  },
});

// The tokens Mantine has no slot for: panel surface, the subtle fill under table heads and footers, muted chart series,
// status colors (with soft fills and lines) and the four categorical series of savings by cause. Charts and custom
// pieces read only these.
const LIGHT: Record<string, string> = {
  "--mo-surface": "#FFFFFF", "--mo-bg-2": "#F1F1F5", "--mo-fg-2": "#3B3E4A", "--mo-line": "#E7E7EC", "--mo-line-strong": "#D6D6DE",
  "--mo-accent": "#4F46E5", "--mo-accent-text": "#4338CA", "--mo-accent-soft": "#EEEDFD", "--mo-accent-line": "#C9C6F8",
  "--mo-bar": "#C4C6D2", "--mo-bar-2": "#E1E2E9",
  "--mo-good": "#167A3E", "--mo-good-soft": "#EAF6EE", "--mo-good-line": "#BCE3C9",
  "--mo-warn": "#9A5B00", "--mo-warn-soft": "#FFF6E6", "--mo-warn-line": "#F0D6A8",
  "--mo-bad": "#B42318", "--mo-bad-soft": "#FDEEEC", "--mo-bad-line": "#F3C5BF",
  "--mo-s1": "#4F46E5", "--mo-s2": "#0E9AA7", "--mo-s3": "#9C97F0", "--mo-s4": "#D9709A",
};
const DARK: Record<string, string> = {
  "--mo-surface": "#14151B", "--mo-bg-2": "#111217", "--mo-fg-2": "#C3C6D2", "--mo-line": "#23252E", "--mo-line-strong": "#30333F",
  "--mo-accent": "#8C88FF", "--mo-accent-text": "#A9A6FF", "--mo-accent-soft": "#1C1B36", "--mo-accent-line": "#3B3880",
  "--mo-bar": "#3E4150", "--mo-bar-2": "#2A2C36",
  "--mo-good": "#5CCB8A", "--mo-good-soft": "#0F1F16", "--mo-good-line": "#1E4A30",
  "--mo-warn": "#F2B860", "--mo-warn-soft": "#211A0E", "--mo-warn-line": "#5A4220",
  "--mo-bad": "#F58B80", "--mo-bad-soft": "#26110F", "--mo-bad-line": "#5C2621",
  "--mo-s1": "#7C77F2", "--mo-s2": "#13979F", "--mo-s3": "#5E58CC", "--mo-s4": "#C55E8B",
};

// Mantine's light and outline variants take a tuple's palest step for text in dark mode, and its darkest in light mode:
// on these tuples that reads as near-white and near-black. Point them at the status and accent tokens instead.
const VARIANT_TOKENS: Record<string, [string, string, string]> = { mo: ["accent-soft", "accent-text", "accent-line"], green: ["good-soft", "good", "good-line"],
  yellow: ["warn-soft", "warn", "warn-line"], red: ["bad-soft", "bad", "bad-line"], gray: ["bg-2", "fg-2", "line-strong"] };
const variants = (s: Record<string, string>) => Object.fromEntries(Object.entries(VARIANT_TOKENS).flatMap(([c, [soft, text, line]]) => [
  [`--mantine-color-${c}-light`, s[`--mo-${soft}`]],
  [`--mantine-color-${c}-light-hover`, `color-mix(in srgb, ${s[`--mo-${line}`]} 45%, ${s[`--mo-${soft}`]})`],
  [`--mantine-color-${c}-light-color`, s[`--mo-${text}`]],
  [`--mantine-color-${c}-outline`, s[`--mo-${text}`]],
  [`--mantine-color-${c}-text`, s[`--mo-${text}`]],
]));

export const cssVariablesResolver: CSSVariablesResolver = () => ({
  variables: {},
  light: {
    ...LIGHT,
    ...variants(LIGHT),
    "--mantine-color-body": "#F7F7F9",
    "--mantine-color-text": "#0E0F14",
    "--mantine-color-dimmed": "#62677A",
    "--mantine-color-default-border": "#E7E7EC",
    "--mantine-color-default-hover": "#F1F1F5",
  },
  dark: {
    ...DARK,
    ...variants(DARK),
    "--mantine-color-body": "#0B0C0F",
    "--mantine-color-text": "#EDEEF3",
    "--mantine-color-dimmed": "#9A9EAE",
    "--mantine-color-default-border": "#23252E",
    "--mantine-color-default-hover": "#1A1B22",
    "--mantine-color-default": "#14151B",
  },
});
