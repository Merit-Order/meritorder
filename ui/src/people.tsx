/** What to call people: their name, else their email address, never an identity provider's GUID when something better is
 * known. Loaded once for the console (GET /v1/reports/people, scoped to who the viewer may see) and shared by every page. */
import { createContext, useContext } from "react";

export type PeopleMap = Record<string, { name: string | null; email: string | null }>;

export const PeopleContext = createContext<PeopleMap>({});

export function personLabel(people: PeopleMap, id: string | null | undefined): string {
  if (!id) return "–";
  const p = people[id];
  return p?.name || p?.email || id;
}

/** A function from a person's id to what to show, for places that need a string (chart labels, messages). */
export function usePersonLabel(): (id: string | null | undefined) => string {
  const people = useContext(PeopleContext);
  return (id) => personLabel(people, id);
}

/** A person, shown by name or email; hovering shows the email and the id behind it. */
export function Person({ id }: { id: string | null | undefined }) {
  const people = useContext(PeopleContext);
  const label = personLabel(people, id);
  const p = id ? people[id] : undefined;
  const title = p ? [p.email !== label ? p.email : null, id !== label ? id : null].filter(Boolean).join(" · ") : undefined;
  return <span title={title || undefined}>{label}</span>;
}
