import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import App from "./App";
import "./style.css";
import "./tester.css";
createRoot(document.getElementById("root")!).render(<App />);
