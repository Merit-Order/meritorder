/// <reference types="vite/client" />

/** True in the demo build (`vite build --mode demo`): the console answers its own API calls from sample data (src/demo/).
 * A constant Vite writes in, so the production build drops every branch that needs it, and the demo with them. */
declare const __DEMO__: boolean;
