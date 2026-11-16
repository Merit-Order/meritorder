import { AppShell, Burger, Button, Group, Menu, NavLink, ScrollArea, Text, UnstyledButton, useMantineColorScheme } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Api, ApiError } from "./api";
import { useData } from "./hooks";
import { Keys, MyUsage, SignIn, Teams } from "./pages/Admin";
import { ConfigPage } from "./pages/config/ConfigPage";
import { People } from "./pages/People";
import { Dispatch } from "./pages/Dispatch";
import { Backends, Models } from "./pages/Infra";
import { Gpus } from "./pages/Gpus";
import { Controls } from "./pages/Controls";
import { Reports } from "./pages/Reports";
import { SourceFooter } from "./components/SourceFooter";
import { BrandMark, Icon } from "./components/icons";
import { DEMO_KEY, DemoBanner } from "./demo/Banner";

const KEY_STORAGE = "meritorder.key";

function readKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

type Page = { id: string; label: string; roles: string[] };
const PAGES: Page[] = [
  { id: "dispatch", label: "Dispatch", roles: ["admin", "finance", "team_lead"] },
  { id: "reports", label: "Reports", roles: ["admin", "finance", "team_lead", "user"] },
  { id: "gpus", label: "GPUs", roles: ["admin", "finance", "team_lead"] },
  { id: "controls", label: "Controls", roles: ["admin", "finance"] },
  { id: "backends", label: "Backends", roles: ["admin", "team_lead"] },
  { id: "models", label: "Models", roles: ["admin"] },
  { id: "teams", label: "Teams", roles: ["admin", "finance", "team_lead"] },
  { id: "people", label: "People", roles: ["admin", "team_lead"] },
  { id: "keys", label: "Keys", roles: ["admin", "team_lead"] },
  { id: "config", label: "Config", roles: ["admin"] },
  { id: "me", label: "My usage", roles: ["admin", "finance", "team_lead", "user"] },
];

/** A text toggle ("Light" / "Dark": the scheme it switches to) rather than a glyph. */
function SchemeToggle() {
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const dark = colorScheme !== "light";
  return (
    <Button variant="default" size="compact-sm" h={32} px={10} leftSection={<Icon name="theme" />}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} onClick={() => setColorScheme(dark ? "light" : "dark")}>
      <Text span inherit visibleFrom="xs">{dark ? "Light" : "Dark"}</Text>
    </Button>
  );
}

export default function App() {
  const [key, setKey] = useState(() => (__DEMO__ ? DEMO_KEY : readKey()));  // the demo signs in as its admin
  const api = useMemo(() => new Api(key), [key]);
  const me = useData(() => (key ? api.me() : Promise.reject(new ApiError(401, "signed out"))), [api]);
  const [page, setPage] = useState(() => location.hash.slice(1));
  const [opened, { toggle, close }] = useDisclosure(false);

  useEffect(() => {
    const onHash = () => setPage(location.hash.slice(1));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = useCallback((id: string) => {
    location.hash = id;
    close();
  }, [close]);
  const saveKey = (k: string) => {
    try { localStorage.setItem(KEY_STORAGE, k); } catch { /* private mode: keep it in memory */ }
    setKey(k);
  };
  const signOut = () => {
    try { localStorage.removeItem(KEY_STORAGE); } catch { /* ignore */ }
    setKey("");
  };

  if (!key || (me.error && !me.data)) return <SignIn onKey={saveKey} error={key ? me.error : null} />;
  if (!me.data) return null;
  const role = me.data.role;
  const pages = PAGES.filter((p) => p.roles.includes(role));
  const active = pages.find((p) => p.id === page.split("/")[0])?.id ?? pages[0].id;  // "models/evals": a tab within a page
  // The current page in the accent; the rest as quiet text buttons. Below 1240px (eleven admin pages) they fold into
  // the burger's side panel.
  const nav = (vertical: boolean) => pages.map((p) => vertical
    ? <NavLink key={p.id} label={p.label} active={active === p.id} onClick={() => go(p.id)} />
    : <Button key={p.id} size="compact-sm" h={30} px={10} fw={active === p.id ? 500 : 400} variant={active === p.id ? "light" : "subtle"}
        color={active === p.id ? "mo" : "gray"} c={active === p.id ? undefined : "var(--mo-fg-2)"} aria-current={active === p.id ? "page" : undefined}
        onClick={() => go(p.id)}>{p.label}</Button>);

  return (
    <AppShell header={{ height: 56 }} navbar={{ width: 220, breakpoint: 1240, collapsed: { desktop: true, mobile: !opened } }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" gap={20} wrap="nowrap" maw={1480} mx="auto">
          <Burger opened={opened} onClick={toggle} className="nav-narrow" size="sm" aria-label="Menu" />
          <UnstyledButton onClick={() => go(pages[0].id)} className="brand" aria-label="Merit Order console home">
            <BrandMark /><span>Merit Order</span>
          </UnstyledButton>
          <Group gap={2} className="nav-wide" wrap="nowrap">{nav(false)}</Group>
          <Group gap={10} ml="auto" wrap="nowrap">
            <span className="live hide-sm" title="Data refreshes on its own"><span className="live-dot" aria-hidden="true" />live</span>
            <SchemeToggle />
            <Menu position="bottom-end" withinPortal>
              <Menu.Target>
                <Button variant="default" size="compact-sm" h={32} px={10} leftSection={<span className="avatar" aria-hidden="true">{me.data.user.slice(0, 1).toUpperCase()}</span>}
                  rightSection={<Icon name="chev" size={12} />}>{me.data.user}</Button>
              </Menu.Target>
              <Menu.Dropdown>
                <Menu.Label>{me.data.team} · {role.replace("_", " ")}</Menu.Label>
                <Menu.Item onClick={() => go("me")}>My usage</Menu.Item>
                <Menu.Item color="red" onClick={signOut}>Sign out</Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Navbar p="xs"><ScrollArea>{nav(true)}</ScrollArea></AppShell.Navbar>
      <AppShell.Main>
        {__DEMO__ && <DemoBanner />}
        <div style={{ maxWidth: 1480, margin: "0 auto", paddingTop: 6 }}>
          {active === "dispatch" && <Dispatch api={api} me={me.data} onNavigate={go} />}
          {active === "reports" && <Reports api={api} me={me.data} />}
          {active === "gpus" && <Gpus api={api} me={me.data} />}
          {active === "controls" && <Controls api={api} me={me.data} />}
          {active === "backends" && <Backends api={api} />}
          {active === "models" && <Models api={api} me={me.data} />}
          {active === "teams" && <Teams api={api} me={me.data} onNavigate={go} />}
          {active === "keys" && <Keys api={api} me={me.data} />}
          {active === "config" && <ConfigPage api={api} />}
          {active === "people" && <People api={api} me={me.data} />}
          {active === "me" && <MyUsage me={me.data} />}
        </div>
        <SourceFooter />
      </AppShell.Main>
    </AppShell>
  );
}
