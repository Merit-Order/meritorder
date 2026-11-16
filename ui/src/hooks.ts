import { useCallback, useEffect, useRef, useState } from "react";

/** Load data, optionally re-polling every `intervalMs` while the tab is visible.
 *
 * A response is used only if it's the newest one asked for under the current `deps`: when they change, what was loaded
 * for the old ones is cleared, and a slow answer to an earlier request (or to the old deps) never replaces a newer one. */
export function useData<T>(fn: () => Promise<T>, deps: unknown[], intervalMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const generation = useRef(0); // bumped when deps change
  const asked = useRef(0); // requests started
  const applied = useRef(0); // the newest request whose answer was used
  const load = useCallback(async () => {
    const gen = generation.current;
    const n = ++asked.current;
    const current = () => gen === generation.current && n > applied.current;
    try {
      const d = await fnRef.current();
      if (!current()) return;
      applied.current = n;
      setData(d);
      setError(null);
    } catch (e) {
      if (!current()) return;
      applied.current = n;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (gen === generation.current && n === asked.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    generation.current += 1;
    applied.current = asked.current;
    setData(null);
    setError(null);
    setLoading(true);
    void load();
    if (!intervalMs) return;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading, reload: load };
}
