import { Alert, Anchor, Button, Code, Group, Modal, Paper, PasswordInput, Stack, Text, TextInput, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState, type ReactNode } from "react";
import { auth } from "../api";
import { BrandMark, Icon } from "../components/icons";
import { SourceFooter } from "../components/SourceFooter";
import { useData } from "../hooks";

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function Frame({ title, intro, children, footer }: { title: string; intro: ReactNode; children: ReactNode; footer?: ReactNode }) {
  return (
    <Stack maw={400} mx="auto" pt="14vh" px="md" gap="lg" mih="100vh">
      <Group gap={10} justify="center">
        <BrandMark size={28} />
        <Title order={1} fz={22} style={{ letterSpacing: "-.02em" }}>Merit Order</Title>
      </Group>
      <Paper px="lg" py="lg">
        <Stack gap="md">
          <div>
            <Text fw={600} size="md">{title}</Text>
            <Text size="sm" c="dimmed" mt={2}>{intro}</Text>
          </div>
          {children}
        </Stack>
      </Paper>
      {footer}
      <div style={{ marginTop: "auto" }}><SourceFooter /></div>
    </Stack>
  );
}

/** The console's sign-in page: email and password (local accounts), or an API key, which the gateway trades for a session
 * so the browser never keeps it. */
export function SignIn({ onSignedIn, notice }: { onSignedIn: () => void; notice?: string | null }) {
  const cfg = useData(() => auth.config(), []);
  const local = cfg.data ? cfg.data.local?.enabled !== false : true;
  const [withKey, setWithKey] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const useKey = withKey || !local;
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await auth.login(useKey ? { api_key: key.trim() } : { email: email.trim(), password });
      setPassword("");
      setKey("");
      onSignedIn();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Frame title="Sign in to the console" intro="The control panel for your organization's LLM traffic."
      footer={<Text size="xs" c="dimmed" ta="center">{useKey
        ? <>Run <Code>meritorder token</Code> to print your key, or <Code>meritorder login</Code> to get one.</>
        : "No account yet? Ask your Merit Order admin for an invite link."}</Text>}>
      <form onSubmit={submit}>
        <Stack gap="sm">
          {useKey ? (
            <PasswordInput label="API key" placeholder="mo-…" value={key} onChange={(e) => setKey(e.currentTarget.value)} autoComplete="off"
              styles={{ input: { fontFamily: "var(--mantine-font-family-monospace)" } }} data-autofocus />
          ) : (
            <>
              <TextInput label="Email address" type="email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} autoComplete="username" required />
              <PasswordInput label="Password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} autoComplete="current-password" required />
            </>
          )}
          <Button type="submit" fullWidth loading={busy} disabled={useKey ? !key.trim() : !email.trim() || !password}>Sign in</Button>
        </Stack>
      </form>
      {(error || notice) && <Alert color="red" icon={<Icon name="bad" size={16} />}>{error ?? notice}</Alert>}
      {local && (
        <Text size="sm" ta="center">
          <Anchor component="button" type="button" size="sm" onClick={() => { setWithKey(!withKey); setError(null); }}>
            {withKey ? "Sign in with your email address instead" : "Sign in with an API key instead"}
          </Anchor>
        </Text>
      )}
    </Frame>
  );
}

/** Where an invite or password-reset link lands (/ui/#invite=<token>): set a password, then you're signed in. */
export function AcceptInvite({ token, onDone }: { token: string; onDone: () => void }) {
  const info = useData(() => auth.invite(token), [token]);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const min = info.data?.min_password_length ?? 12;
  const reset = info.data?.kind === "reset";
  const mismatch = again.length > 0 && again !== password;
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await auth.acceptInvite(token, { password, ...(name.trim() && !reset ? { name: name.trim() } : {}) });
      history.replaceState(null, "", location.pathname);  // the link worked once; don't leave it in the address bar
      onDone();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  const leave = () => { history.replaceState(null, "", location.pathname); onDone(); };
  if (info.error) {
    return (
      <Frame title="This link doesn't work" intro={info.error}>
        <Button variant="default" onClick={leave}>Go to sign-in</Button>
      </Frame>
    );
  }
  return (
    <Frame title={reset ? "Set a new password" : "Welcome to Merit Order"}
      intro={info.data ? <>{reset ? "For " : "You've been invited as "}<b>{info.data.email}</b>
        {!reset && info.data.team ? <> (team {info.data.team}, {info.data.role?.replace("_", " ")})</> : null}.
        {" "}Choose a password of at least {min} characters; a few unrelated words work well.</> : "Checking the link…"}>
      <form onSubmit={submit}>
        <Stack gap="sm">
          <input type="email" value={info.data?.email ?? ""} autoComplete="username" readOnly hidden />
          {!reset && <TextInput label="Your name" placeholder="optional" value={name} onChange={(e) => setName(e.currentTarget.value)} autoComplete="name" />}
          <PasswordInput label="Password" value={password} onChange={(e) => setPassword(e.currentTarget.value)} autoComplete="new-password" required
            description={password && password.length < min ? `${min - password.length} more characters` : undefined} />
          <PasswordInput label="Again" value={again} onChange={(e) => setAgain(e.currentTarget.value)} autoComplete="new-password" required
            error={mismatch ? "doesn't match" : undefined} />
          <Button type="submit" fullWidth loading={busy} disabled={!info.data || password.length < min || again !== password}>
            {reset ? "Set password and sign in" : "Create account and sign in"}
          </Button>
        </Stack>
      </form>
      {error && <Alert color="red" icon={<Icon name="bad" size={16} />}>{error}</Alert>}
    </Frame>
  );
}

/** Change your own password (account menu). Your other sessions end; this one stays. */
export function ChangePassword({ opened, onClose }: { opened: boolean; onClose: () => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const close = () => { setCurrent(""); setNext(""); setAgain(""); setError(null); onClose(); };
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const out = await auth.changePassword(current, next);
      notifications.show({ color: "green", title: "Password changed",
        message: out.other_sessions_ended ? `Signed out of ${out.other_sessions_ended} other session${out.other_sessions_ended === 1 ? "" : "s"}.` : "This session stays signed in." });
      close();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal opened={opened} onClose={close} title="Change your password">
      <form onSubmit={submit}>
        <Stack gap="sm">
          <PasswordInput label="Current password" value={current} onChange={(e) => setCurrent(e.currentTarget.value)} autoComplete="current-password" required data-autofocus />
          <PasswordInput label="New password" value={next} onChange={(e) => setNext(e.currentTarget.value)} autoComplete="new-password" required />
          <PasswordInput label="Again" value={again} onChange={(e) => setAgain(e.currentTarget.value)} autoComplete="new-password" required
            error={again && again !== next ? "doesn't match" : undefined} />
          {error && <Alert color="red" icon={<Icon name="bad" size={16} />}>{error}</Alert>}
          <Group justify="flex-end"><Button variant="default" onClick={close}>Cancel</Button>
            <Button type="submit" loading={busy} disabled={!current || !next || again !== next}>Change password</Button></Group>
        </Stack>
      </form>
    </Modal>
  );
}
