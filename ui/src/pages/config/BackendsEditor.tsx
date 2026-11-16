import { StatusBadge } from "../../components/common";
import { Alert, Badge, Button, Divider, Drawer, Group, Select, SimpleGrid, Stack, Switch, Table, TagsInput, Text, TextInput } from "@mantine/core";
import { useState } from "react";
import type { BackendStatus, Doc } from "../../api";
import { EnvName, Flag, Num, SectionTitle, Txt, lbl, rich, setIn } from "./fields";
import { BACKEND as H, SECTION } from "./help";

const AUTH_TYPES = [
  { value: "none", label: "None" },
  { value: "api_key", label: "API key (from an environment variable)" },
  { value: "aws_sigv4", label: "AWS (SigV4 or Bedrock bearer token)" },
  { value: "azure_ad", label: "Microsoft Entra (managed identity / az login)" },
  { value: "google", label: "Google (application default credentials)" },
];

function BackendForm({ b, models, onChange }: { b: Doc; models: string[]; onChange: (b: Doc) => void }) {
  const set = (path: (string | number)[], v: unknown) => onChange(setIn(b, path, v));
  const kind = b.kind ?? "api";
  const auth: Doc = b.auth ?? {};
  const authType = auth.type ?? "none";
  return (
    <Stack gap="md">
      <SimpleGrid cols={2}>
        <Flag label="Enabled" value={b.enabled} onChange={(v) => set(["enabled"], v)} description="Also switches off by itself when its credentials are missing." help={H.enabled} />
        <Select {...lbl("Kind", H.kind)} allowDeselect={false} value={kind} onChange={(v) => set(["kind"], v === "api" ? undefined : v)}
          data={[{ value: "api", label: "Provider API" }, { value: "self_hosted", label: "Self-hosted (vLLM / SGLang)" }, { value: "mock", label: "Mock (testing)" }]} />
        <Select {...lbl("Wire format", H.format)} allowDeselect={false} value={b.format ?? "openai"} onChange={(v) => set(["format"], v)}
          data={[{ value: "anthropic", label: "Anthropic Messages" }, { value: "openai", label: "OpenAI-compatible" }]} />
        {b.format !== "anthropic" && kind !== "mock" && (
          <Select {...lbl("Endpoint", H.api)} allowDeselect={false} value={b.api ?? "chat"}
            onChange={(v) => onChange(setIn(setIn(setIn(b, ["api"], v === "chat" ? undefined : v), ["flavor"], v === "gemini" ? b.flavor : undefined),
              ["responses_store"], v === "responses" ? b.responses_store : undefined))}
            data={[{ value: "chat", label: "Chat Completions (/chat/completions)" }, { value: "responses", label: "Responses (/responses)" },
              { value: "gemini", label: "Gemini native (generateContent)" }]} />
        )}
        {(b.format === "anthropic" || b.api === "gemini") && (
          <Select {...lbl("Flavor", H.flavor)} allowDeselect={false} value={b.flavor ?? "standard"} onChange={(v) => set(["flavor"], v === "standard" ? undefined : v)}
            data={[{ value: "standard", label: b.api === "gemini" ? "Standard (the Gemini API)" : "Standard (Anthropic, Bedrock, Foundry)" },
              { value: "vertex", label: "Vertex AI" }]} />
        )}
      </SimpleGrid>
      {kind !== "mock" && b.flavor !== "vertex" && (
        <Txt label="Base URL" value={b.base_url} onChange={(v) => set(["base_url"], v)} mono help={H.base_url}
          placeholder={b.format === "anthropic" ? "https://api.anthropic.com" : b.api === "gemini" ? "https://generativelanguage.googleapis.com/v1beta" : "https://openrouter.ai/api/v1"}
          description="May use ${ENV} references. For self-hosted backends started by SkyPilot, leave blank." />
      )}
      {b.flavor === "vertex" && (
        <SimpleGrid cols={2}>
          <Txt label="GCP project" value={b.project} onChange={(v) => set(["project"], v)} placeholder="${GOOGLE_CLOUD_PROJECT}" mono help={H.project} />
          <Txt label="Region" value={b.region} onChange={(v) => set(["region"], v)} placeholder="global" help={H.vertex_region} />
        </SimpleGrid>
      )}
      {kind !== "mock" && (
        <>
          <Divider />
          <SectionTitle title="Credentials" />
          <Select {...lbl("How Merit Order signs in", H.auth)} data={AUTH_TYPES} allowDeselect={false} value={authType}
            onChange={(v) => set(["auth"], v === "none" ? undefined : { type: v })} />
          {authType === "api_key" && (
            <SimpleGrid cols={3}>
              <EnvName label="Key environment variable" value={auth.value_env} onChange={(v) => set(["auth", "value_env"], v)} description="e.g. OPENROUTER_API_KEY" help={H.value_env} />
              <Txt label="Header" value={auth.header} onChange={(v) => set(["auth", "header"], v)} placeholder="authorization" description="x-api-key for Anthropic-style" help={H.header} />
              <Txt label="Prefix" value={auth.prefix} onChange={(v) => set(["auth", "prefix"], v)} placeholder='"Bearer " for authorization' help={H.prefix} />
              <Txt label="Key file (optional)" value={auth.value_file} onChange={(v) => set(["auth", "value_file"], v)} mono
                placeholder="/run/secrets/meritorder/OPENROUTER_API_KEY" help={H.value_file}
                description="A mounted secret. Re-read when it changes, so a rotated key works without a restart; falls back to the variable." />
            </SimpleGrid>
          )}
          {authType === "aws_sigv4" && (
            <SimpleGrid cols={3}>
              <Txt label="Service" value={auth.service} onChange={(v) => set(["auth", "service"], v)} placeholder="bedrock-mantle" help={H.aws_service} />
              <Txt label="Region" value={auth.region} onChange={(v) => set(["auth", "region"], v)} placeholder="${AWS_REGION:-us-east-1}" mono help={H.aws_region} />
              <Txt label="Profile" value={auth.profile} onChange={(v) => set(["auth", "profile"], v)} placeholder="${AWS_PROFILE:-}" mono help={H.aws_profile} />
            </SimpleGrid>
          )}
          {authType === "azure_ad" && <Txt label="Token scope" value={auth.scope} onChange={(v) => set(["auth", "scope"], v)} placeholder="https://cognitiveservices.azure.com/.default" help={H.azure_scope} />}
          {authType !== "none" && <Text size="xs" c="dimmed">Secrets are read from the gateway's environment at runtime and never stored in the configuration.</Text>}
        </>
      )}
      <Divider />
      <SectionTitle title="Behaviour" />
      <SimpleGrid cols={2}>
        <Select {...lbl("Reasoning effort parameter", H.effort_format)} allowDeselect={false} value={b.effort_format ?? "auto"} onChange={(v) => set(["effort_format"], v === "auto" ? undefined : v)}
          data={["auto", "anthropic", "openai", "openrouter", "none"]} description="auto: detected from the format and URL." />
        <Num label="Prompt cache lifetime (seconds)" value={b.cache_ttl_seconds} onChange={(v) => set(["cache_ttl_seconds"], v)} min={0} placeholder="300"
          description="Default 300; 3600 asks Claude for 1-hour caching from a session's second turn." help={H.cache_ttl_seconds} />
        <Num label="Timeout (seconds)" value={b.timeout_seconds} onChange={(v) => set(["timeout_seconds"], v)} min={1} placeholder="600" help={H.timeout_seconds} />
        <Num label="Give up if a stream hasn't started (seconds)" value={b.first_byte_timeout_seconds} onChange={(v) => set(["first_byte_timeout_seconds"], v)} min={1}
          placeholder={kind === "self_hosted" ? "off" : "120"} help={H.first_byte_timeout_seconds} />
        {b.format === "openai" && <Select {...lbl("Max tokens field", H.max_tokens_field)} value={b.max_tokens_field ?? "max_tokens"} allowDeselect={false}
          onChange={(v) => set(["max_tokens_field"], v === "max_tokens" ? undefined : v)} data={["max_tokens", "max_completion_tokens"]} />}
        {b.kind !== "self_hosted" && <Select {...lbl("Recorded cost", H.billed_cost)} allowDeselect={false} value={b.billed_cost ?? "auto"}
          onChange={(v) => set(["billed_cost"], v === "auto" ? undefined : v)} data={["auto", "reported", "catalog"]}
          description="reported: what the backend says it billed (OpenRouter's usage.cost). auto: reported for OpenRouter." />}
        {b.format === "anthropic" && <Txt label="anthropic-version header" value={b.anthropic_version} onChange={(v) => set(["anthropic_version"], v)} placeholder="client's value" help={H.anthropic_version} />}
      </SimpleGrid>
      {b.format === "anthropic" && <Switch {...lbl("Forward the client's anthropic-beta header", H.forward_anthropic_beta)} checked={b.forward_anthropic_beta !== false}
        onChange={(e) => set(["forward_anthropic_beta"], e.currentTarget.checked ? undefined : false)} />}
      {b.format === "openai" && <Switch {...lbl("Send prompt_cache_key (OpenAI)", H.supports_prompt_cache_key)} checked={!!b.supports_prompt_cache_key}
        onChange={(e) => set(["supports_prompt_cache_key"], e.currentTarget.checked || undefined)} />}
      {b.format === "openai" && b.api === "responses" && (
        <Switch {...lbl("Let the provider store stored-mode responses too", H.responses_store)} checked={!!b.responses_store}
          description="Only for teams with stored responses on. The provider then also holds their content, for its own retention (OpenAI: 30 days)."
          onChange={(e) => set(["responses_store"], e.currentTarget.checked || undefined)} />
      )}
      {kind === "self_hosted" && (
        <>
          <Divider />
          <SectionTitle title="GPUs" intro={SECTION.gpus} />
          <SimpleGrid cols={2}>
            <Num label="Cost per replica per hour" value={b.gpu_cost_per_hour} onChange={(v) => set(["gpu_cost_per_hour"], v)} min={0} prefix="$" help={H.gpu_cost_per_hour} />
            <Num label="Tokens/s one replica sustains" value={b.capacity_tokens_per_second} onChange={(v) => set(["capacity_tokens_per_second"], v)} min={1} placeholder="2,000"
              description="Measured at your latency target on real traffic." help={H.capacity_tokens_per_second} />
            <Num label="Requests one replica runs at once" value={b.max_concurrency} onChange={(v) => set(["max_concurrency"], v)} min={1} placeholder="not set"
              description="vLLM --max-num-seqs, or what the KV cache fits. Set it so gateways see a full GPU within seconds." help={H.max_concurrency} />
            <Num label="Requests allowed to wait per replica" value={b.max_waiting} onChange={(v) => set(["max_waiting"], v)} min={0} placeholder="32"
              description="When this GPU is full and nothing else serves the model, requests wait in vLLM up to this many; then 503." help={H.max_waiting} />
          </SimpleGrid>
          <Switch {...lbl("Let Merit Order start and stop it", H.serve)} description={SECTION.serve} checked={!!b.serve}
            onChange={(e) => set(["serve"], e.currentTarget.checked ? { provisioner: "skypilot", model: models[0] } : undefined)} />
          {b.serve && (
            <Stack gap="sm">
              <SimpleGrid cols={2}>
                <Select {...lbl("Provisioner", H.provisioner)} allowDeselect={false} value={b.serve.provisioner ?? "skypilot"} onChange={(v) => set(["serve", "provisioner"], v)}
                  data={[{ value: "skypilot", label: "SkyPilot (cloud GPUs)" }, { value: "kubernetes", label: "Kubernetes (your cluster: KServe or llm-d)" },
                    { value: "static", label: "Static (a server you already run)" }]} />
                <Select {...lbl("Model it serves", H.serve_model)} data={models} searchable value={b.serve.model ?? null} onChange={(v) => set(["serve", "model"], v ?? undefined)} />
              </SimpleGrid>
              {b.serve.provisioner === "static" ? (
                <>
                  <SimpleGrid cols={2}>
                    <Txt label="Server URL (…/v1)" value={b.serve.static_url} onChange={(v) => set(["serve", "static_url"], v)} mono help={H.static_url} />
                    <Txt label="Where it runs" value={b.serve.location} onChange={(v) => set(["serve", "location"], v)} placeholder="on-prem, rack 4" help={H.location} />
                  </SimpleGrid>
                  <TagsInput {...lbl("Replica URLs (…/v1 each)", H.static_urls)} description="Optional. Each replica's own address, so a session goes back to the replica that holds its cache."
                    value={b.serve.static_urls ?? []} onChange={(v) => set(["serve", "static_urls"], v.length ? v : undefined)} />
                </>
              ) : (
                <>
                  <SimpleGrid cols={3}>
                    <Txt label="Accelerators" value={b.serve.accelerators} onChange={(v) => set(["serve", "accelerators"], v)} placeholder="H100:8" mono help={H.accelerators} />
                    <Txt label="vLLM image" value={b.serve.image} onChange={(v) => set(["serve", "image"], v)} placeholder="vllm/vllm-openai:v0.32.0-cu129" mono help={H.image} />
                    <Switch mt="xl" {...lbl("Use spot instances", H.use_spot)} checked={!!b.serve.use_spot} onChange={(e) => set(["serve", "use_spot"], e.currentTarget.checked || undefined)} />
                  </SimpleGrid>
                  {b.serve.provisioner === "kubernetes" ? (
                    <SimpleGrid cols={2}>
                      <Select {...lbl("Runs through", H.k8s_api)} allowDeselect={false} value={b.serve.kubernetes?.api ?? "kserve"}
                        onChange={(v) => set(["serve", "kubernetes", "api"], v)}
                        data={[{ value: "kserve", label: "KServe (an InferenceService)" }, { value: "llm-d", label: "llm-d (behind its InferencePool)" }]} />
                      <Txt label="Namespace" value={b.serve.kubernetes?.namespace} onChange={(v) => set(["serve", "kubernetes", "namespace"], v)}
                        placeholder="the cluster default" mono help={H.k8s_namespace} />
                      <Txt label="Node type" value={b.serve.kubernetes?.node_type} onChange={(v) => set(["serve", "kubernetes", "node_type"], v)}
                        placeholder="p5.48xlarge" mono help={H.k8s_node_type} />
                      <Txt label="CPUs a replica may use" value={b.serve.kubernetes?.cpu} onChange={(v) => set(["serve", "kubernetes", "cpu"], v)}
                        placeholder={b.serve.kubernetes?.api === "llm-d" ? "no limit" : "required, e.g. 32"} mono help={H.k8s_resources} />
                      <Txt label="Memory a replica may use" value={b.serve.kubernetes?.memory} onChange={(v) => set(["serve", "kubernetes", "memory"], v)}
                        placeholder={b.serve.kubernetes?.api === "llm-d" ? "no limit" : "required, e.g. 256Gi"} mono help={H.k8s_resources} />
                      <Txt label="Hugging Face token Secret" value={b.serve.kubernetes?.hf_token_secret} onChange={(v) => set(["serve", "kubernetes", "hf_token_secret"], v)}
                        placeholder="none" mono help={H.k8s_hf_token_secret} />
                      <Txt label="Endpoint (…/v1)" value={b.serve.kubernetes?.endpoint} onChange={(v) => set(["serve", "kubernetes", "endpoint"], v)}
                        placeholder={b.serve.kubernetes?.api === "llm-d" ? "http://llm-d-gateway.llm-d.svc.cluster.local/v1" : "KServe's address"} mono help={H.k8s_endpoint} />
                      {b.serve.kubernetes?.api === "llm-d" && <Txt label="InferencePool" value={b.serve.kubernetes?.inference_pool}
                        onChange={(v) => set(["serve", "kubernetes", "inference_pool"], v)} mono help={H.k8s_inference_pool} />}
                    </SimpleGrid>
                  ) : (
                    <SimpleGrid cols={2}>
                      <TagsInput {...lbl("Clouds", H.clouds)} description="Blank: SkyPilot picks the cheapest." data={["aws", "gcp", "azure", "lambda", "runpod", "kubernetes"]}
                        value={b.serve.clouds ?? []} onChange={(v) => set(["serve", "clouds"], v.length ? v : undefined)} />
                      <TagsInput {...lbl("Regions", H.regions)} description="Blank: any." value={b.serve.regions ?? []}
                        onChange={(v) => set(["serve", "regions"], v.length ? v : undefined)} />
                      <Num label="Lease (minutes)" value={b.serve.lease_minutes} onChange={(v) => set(["serve", "lease_minutes"], v)} min={3}
                        placeholder="15" help={H.lease_minutes} />
                      <Num label="Delete after idle (minutes)" value={b.serve.autostop_minutes} onChange={(v) => set(["serve", "autostop_minutes"], v)} min={1}
                        placeholder="5" help={H.autostop_minutes} />
                      <Select {...lbl("SSH tunnel", H.ssh_tunnel)} allowDeselect={false} value={b.serve.ssh_tunnel ?? "auto"}
                        onChange={(v) => set(["serve", "ssh_tunnel"], v === "auto" ? undefined : v)}
                        data={[{ value: "auto", label: "Only where the port can't be reached" }, { value: "always", label: "Always" }, { value: "never", label: "Never" }]} />
                      <Num label="Port" value={b.serve.port} onChange={(v) => set(["serve", "port"], v)} min={1} max={65535} placeholder="8000" help={H.serve_port} />
                      <Select {...lbl("Weights cache", H.weights_cache)} allowDeselect={false} value={b.serve.weights_cache ?? "none"}
                        onChange={(v) => set(["serve", "weights_cache"], v === "none" ? undefined : v)}
                        data={[{ value: "none", label: "None: download each start" }, { value: "volume", label: "A volume (Kubernetes, RunPod)" }]} />
                      {b.serve.weights_cache === "volume" && <>
                        <Txt label="Volume where" value={b.serve.weights_cache_infra} onChange={(v) => set(["serve", "weights_cache_infra"], v)}
                          placeholder="runpod/CA/CA-MTL-1" mono help={H.weights_cache_infra} />
                        <Num label="Volume size (GB)" value={b.serve.weights_cache_gb} onChange={(v) => set(["serve", "weights_cache_gb"], v)} min={1}
                          placeholder="required" help={H.weights_cache_gb} />
                        <Num label="Volume $ per GB a month" value={b.serve.weights_cache_usd_per_gb_month} onChange={(v) => set(["serve", "weights_cache_usd_per_gb_month"], v)}
                          min={0} step={0.01} placeholder="RunPod's price" help={H.weights_cache_usd_per_gb_month} />
                      </>}
                    </SimpleGrid>
                  )}
                  <Txt label="Extra vLLM arguments" value={b.serve.vllm_args} onChange={(v) => set(["serve", "vllm_args"], v)} placeholder="--max-model-len 262144" mono help={H.vllm_args} />
                  <Txt label="Load format" value={b.serve.load_format} onChange={(v) => set(["serve", "load_format"], v)} placeholder="auto" mono help={H.load_format} />
                </>
              )}
              <SimpleGrid cols={3}>
                <Num label="Min replicas" value={b.serve.min_replicas} onChange={(v) => set(["serve", "min_replicas"], v)} min={0} placeholder="0" help={H.min_replicas} />
                <Num label="Max replicas" value={b.serve.max_replicas} onChange={(v) => set(["serve", "max_replicas"], v)} min={1} placeholder="4" help={H.max_replicas} />
                <Num label="Target utilization" value={b.serve.target_utilization} onChange={(v) => set(["serve", "target_utilization"], v)} min={0.1} max={1} step={0.05}
                  placeholder="0.7" help={H.target_utilization} />
                <Num label="Start after (minutes) of demand" value={b.serve.scale_up_after_minutes} onChange={(v) => set(["serve", "scale_up_after_minutes"], v)} min={0}
                  placeholder="10" help={H.scale_up_after_minutes} />
                <Num label="Keep at least (minutes)" value={b.serve.hold_minutes} onChange={(v) => set(["serve", "hold_minutes"], v)} min={0} placeholder="30" help={H.hold_minutes} />
                <Num label="Drain for (minutes)" value={b.serve.drain_minutes} onChange={(v) => set(["serve", "drain_minutes"], v)} min={0} placeholder="15" help={H.drain_minutes} />
                <Num label="Break-even margin" value={b.serve.margin} onChange={(v) => set(["serve", "margin"], v)} min={0} max={1} step={0.05} placeholder="0.2" help={H.margin} />
                <Num label="Give up on a launch after (minutes)" value={b.serve.provision_timeout_minutes} onChange={(v) => set(["serve", "provision_timeout_minutes"], v)} min={1}
                  placeholder="45" help={H.provision_timeout_minutes} />
                <Num label="Cold start until measured (minutes)" value={b.serve.cold_start_minutes} onChange={(v) => set(["serve", "cold_start_minutes"], v)} min={0}
                  placeholder="12" help={H.cold_start_minutes} />
              </SimpleGrid>
              <SimpleGrid cols={3}>
                <Switch mt="md" {...lbl("Keep each session on its replica", H.replica_affinity)} checked={b.serve.replica_affinity !== false}
                  disabled={(b.serve.provisioner ?? "skypilot") === "skypilot"}
                  description={(b.serve.provisioner ?? "skypilot") === "skypilot" ? "Always on with SkyPilot: there's no load balancer in front of its replicas." : undefined}
                  onChange={(e) => set(["serve", "replica_affinity"], e.currentTarget.checked ? undefined : false)} />
                <Switch mt="md" {...lbl("Follow vLLM's KV events", H.kv_events)} checked={!!b.serve.kv_events}
                  onChange={(e) => set(["serve", "kv_events"], e.currentTarget.checked || undefined)} />
                {b.serve.kv_events && <Num label="KV events port" value={b.serve.kv_events_port} onChange={(v) => set(["serve", "kv_events_port"], v)} min={1} max={65535}
                  placeholder="5557" help={H.kv_events_port} />}
              </SimpleGrid>
            </Stack>
          )}
        </>
      )}
    </Stack>
  );
}

export function BackendsEditor({ draft, setDraft, modelNames, status }: { draft: Doc; setDraft: (d: Doc) => void; modelNames: string[]; status: BackendStatus[] }) {
  const backends: Doc = draft.backends ?? {};
  const [editing, setEditing] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const live = Object.fromEntries(status.map((s) => [s.name, s]));
  const usedBy = (name: string) => Object.entries(draft.models ?? {}).filter(([, m]) => ((m as Doc).routes ?? []).some((r: Doc) => r.backend === name)).map(([n]) => n);
  const add = () => {
    const n = newName.trim().toLowerCase().replace(/[^a-z0-9_.-]/g, "-");
    if (!n || backends[n]) return;
    setDraft(setIn(draft, ["backends", n], { format: "openai", base_url: "", auth: { type: "api_key", value_env: "" } }));
    setNewName("");
    setEditing(n);
  };
  return (
    <Stack gap="md">
      <Text size="sm" c="dimmed">{rich(SECTION.backends)}</Text>
      <Table.ScrollContainer minWidth={820}>
        <Table highlightOnHover verticalSpacing={5} fz="sm">
          <Table.Thead><Table.Tr><Table.Th>Backend</Table.Th><Table.Th>Status now</Table.Th><Table.Th>Kind</Table.Th><Table.Th>Credentials</Table.Th><Table.Th>Used by</Table.Th><Table.Th /></Table.Tr></Table.Thead>
          <Table.Tbody>
            {Object.keys(backends).map((n) => {
              const b = backends[n] ?? {};
              const s = live[n];
              return (
                <Table.Tr key={n} style={{ cursor: "pointer" }} onClick={() => setEditing(n)}>
                  <Table.Td><Text fw={600} size="sm">{n}</Text><Text size="xs" c="dimmed" className="mono" truncate="end" maw={260}>{b.base_url ?? ""}</Text></Table.Td>
                  <Table.Td>{s ? (s.live ? <StatusBadge kind="good">live</StatusBadge> : <StatusBadge kind="off" title={s.reason ?? ""}>off</StatusBadge>) : <Badge color="mo">new</Badge>}
                    {s && !s.live && s.reason && <Text size="xs" c="dimmed">{s.reason}</Text>}</Table.Td>
                  <Table.Td>{b.kind === "self_hosted" ? "self-hosted" : b.kind === "mock" ? "mock" : `${b.format ?? "openai"} API`}</Table.Td>
                  <Table.Td className="mono" style={{ fontSize: 12 }}>{b.auth?.value_file ?? b.auth?.value_env ?? b.auth?.type ?? "–"}</Table.Td>
                  <Table.Td><Text size="xs" c="dimmed">{usedBy(n).length} models</Text></Table.Td>
                  <Table.Td><Button size="compact-xs" variant="subtle">Edit</Button></Table.Td>
                </Table.Tr>
              );
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Group gap="xs" align="flex-end">
        <TextInput label="New backend" placeholder="e.g. together" value={newName} onChange={(e) => setNewName(e.currentTarget.value)} onKeyDown={(e) => e.key === "Enter" && add()} />
        <Button variant="default" onClick={add} disabled={!newName.trim()}>Add backend</Button>
      </Group>
      <Drawer opened={!!editing} onClose={() => setEditing(null)} position="right" size="xl" title={editing ? `Backend: ${editing}` : ""}>
        {editing && backends[editing] && (
          <Stack>
            {live[editing] && !live[editing].live && live[editing].reason && <Alert color="gray" variant="light">Off right now: {live[editing].reason}</Alert>}
            <BackendForm b={backends[editing]} models={modelNames} onChange={(nb) => setDraft(setIn(draft, ["backends", editing], nb))} />
            <Group justify="space-between" mt="md">
              <Button color="red" variant="outline" disabled={usedBy(editing).length > 0} title={usedBy(editing).length ? `Used by ${usedBy(editing).join(", ")}` : ""}
                onClick={() => { setDraft(setIn(draft, ["backends", editing], undefined)); setEditing(null); }}>Delete backend</Button>
              <Button onClick={() => setEditing(null)}>Done</Button>
            </Group>
          </Stack>
        )}
      </Drawer>
    </Stack>
  );
}
