import { Anchor, Group, Text } from "@mantine/core";

/** The key the demo signs in with. Nothing checks it: the demo answers every call itself. */
export const DEMO_KEY = "mo-demo-admin-key-for-the-browser-only";

/** A line above every page of the demo: what this is, and where the real thing is. */
export function DemoBanner() {
  return (
    <Group justify="center" gap="xs" wrap="wrap" mb="sm" py={4} px="sm" role="note" aria-label="Demo"
      style={{ border: "1px dashed var(--mantine-color-dimmed)", borderRadius: 4 }}>
      <Text size="xs" fw={600}>Demo: sample data</Text>
      <Text size="xs" c="dimmed">An invented organization. Changes last until you reload; nothing is saved or sent.</Text>
      <Anchor size="xs" href="https://meritorder.dev" target="_blank" rel="noreferrer">meritorder.dev</Anchor>
      <Anchor size="xs" href="https://github.com/Merit-Order/meritorder" target="_blank" rel="noreferrer">GitHub</Anchor>
    </Group>
  );
}
