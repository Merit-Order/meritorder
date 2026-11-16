/** The demo's clock: it starts at the moment the sample data was recorded (data.json's `now`) and runs from there, so
 * "this month", "2m ago" and the period presets read as they did then, whenever someone opens the demo. */
export function startClock(recordedAt: string): void {
  const Real = Date;
  const offset = Real.parse(recordedAt) - Real.now();
  class DemoDate extends Real {
    constructor(...args: unknown[]) {
      if (args.length === 0) super(Real.now() + offset);
      // @ts-expect-error: Date's own overloads, passed on as given
      else super(...args);
    }

    static now() {
      return Real.now() + offset;
    }
  }
  globalThis.Date = DemoDate as DateConstructor;
}
