/** data.json, written by scripts/build_demo_data.py: every answer the console asks for, keyed "METHOD /path?sorted=query",
 * and the moment they were recorded. Declared here so TypeScript doesn't work out a type for two megabytes of JSON. */
declare module "*/data.json" {
  const data: { now: string; seed: number; answers: Record<string, unknown> };
  export default data;
}
