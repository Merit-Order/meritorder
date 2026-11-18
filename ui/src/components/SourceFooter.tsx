import { Anchor, Group, Text } from "@mantine/core";
import { useEffect, useState } from "react";

/** GET /about: what's running and where its source is (no key needed). */
export type About = { name: string; version: string; license: string; source_url: string; third_party_notices: string };

// The shipped config's console.source_url; shown until /about answers, or if it can't.
const FALLBACK: About = {
  name: "Merit Order", version: "", license: "AGPL-3.0-only", source_url: "https://github.com/therick/meritorder",
  third_party_notices: "/ui/third-party-notices.txt",
};

let pending: Promise<About> | null = null;

function loadAbout(): Promise<About> {
  pending ??= fetch("/about")
    .then((r) => (r.ok ? (r.json() as Promise<About>) : FALLBACK))
    .catch(() => FALLBACK);
  return pending;
}

/** The software's own line on every screen, signed in or not: its version and license, a link to its source (the AGPL's
 * offer to everyone using it over the network; console.source_url), and the bundled packages' notices. */
export function SourceFooter() {
  const [about, setAbout] = useState<About>(FALLBACK);
  useEffect(() => {
    let live = true;
    loadAbout().then((a) => { if (live) setAbout(a); });
    return () => { live = false; };
  }, []);
  return (
    <Group component="footer" justify="center" gap={8} py="lg" mt="xl" wrap="wrap" maw={1480} mx="auto" style={{ borderTop: "1px solid var(--mo-line)" }}>
      <Text size="xs" c="dimmed">{about.name}{about.version ? ` ${about.version}` : ""} · {about.license}</Text>
      <Text size="xs" c="dimmed">·</Text>
      <Anchor size="xs" c="var(--mo-fg-2)" href={about.source_url} target="_blank" rel="noreferrer">Source</Anchor>
      <Text size="xs" c="dimmed">·</Text>
      <Anchor size="xs" c="var(--mo-fg-2)" href={about.third_party_notices} target="_blank" rel="noreferrer">Third-party notices</Anchor>
    </Group>
  );
}
