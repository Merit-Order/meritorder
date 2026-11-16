/** A small YAML writer for the demo's Config → YAML tab: the configuration document (plain JSON data) as block YAML.
 * The gateway writes YAML with PyYAML; this is close enough to read, and the demo never parses it back. */

type Value = string | number | boolean | null | undefined | Value[] | { [k: string]: Value };

const PLAIN = /^[A-Za-z_./${}][\w ./${}:@*+,()-]*$/;
const RESERVED = /^(true|false|yes|no|on|off|null|~|-?\d[\d_.eE+-]*)$/i;

function scalar(v: Value): string {
  if (v === null || v === undefined) return "null";
  if (typeof v !== "string") return String(v);
  return plain(v) ? v : JSON.stringify(v);
}

const plain = (s: string) => !!s && PLAIN.test(s) && !RESERVED.test(s) && !/: | #|:$/.test(s);
const key = (k: string) => (plain(k) ? k : JSON.stringify(k));

function emit(v: Value, indent: string, out: string[], prefix: string): void {
  if (Array.isArray(v)) {
    if (!v.length) { out.push(`${prefix}[]`); return; }
    if (prefix) out.push(prefix.trimEnd());
    for (const item of v) {
      if (item && typeof item === "object" && !Array.isArray(item) && Object.keys(item).length) {
        const lines: string[] = [];
        emit(item, indent + "  ", lines, "");
        out.push(`${indent}- ${lines[0].trimStart()}`, ...lines.slice(1));
      } else emit(item, indent + "  ", out, `${indent}- `);
    }
    return;
  }
  if (v && typeof v === "object") {
    const entries = Object.entries(v).filter(([, x]) => x !== undefined);
    if (!entries.length) { out.push(`${prefix}{}`); return; }
    if (prefix) out.push(prefix.trimEnd());
    for (const [k, x] of entries) {
      const nested = x !== null && typeof x === "object" && (Array.isArray(x) ? x.length : Object.keys(x).length);
      if (nested) emit(x, indent + "  ", out, `${indent}${key(k)}:`);
      else out.push(`${indent}${key(k)}: ${Array.isArray(x) ? "[]" : x && typeof x === "object" ? "{}" : scalar(x)}`);
    }
    return;
  }
  out.push(`${prefix}${scalar(v)}`);
}

export function toYaml(doc: unknown): string {
  const out: string[] = [];
  emit(doc as Value, "", out, "");
  return out.join("\n") + "\n";
}
