import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "./styles.css";
import { MantineProvider } from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { TipLayer } from "./components/common";
import { cssVariablesResolver, theme } from "./theme";

function start() {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <MantineProvider theme={theme} cssVariablesResolver={cssVariablesResolver} defaultColorScheme="dark">
        <Notifications position="top-right" />
        <App />
        <TipLayer />
      </MantineProvider>
    </StrictMode>,
  );
}

// The demo build answers the console's API calls in the browser, from sample data (src/demo/). The production build
// has no demo code: __DEMO__ is false there, so this branch and the import go.
if (__DEMO__) void import("./demo").then((demo) => demo.install()).then(start);
else start();
