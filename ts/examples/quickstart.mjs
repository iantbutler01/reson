// Run from the package dir after `npm run build`:
//   ANTHROPIC_API_KEY=... node examples/quickstart.mjs
// or against any OpenAI-compatible server via a custom-openai model string.
//
// In your own project this import is `from "chevalier"`.
import { Runtime, agentic } from "../index.js";
import { z } from "zod";

const MODEL = process.env.CHEVALIER_MODEL ?? "anthropic:claude-3-5-sonnet";
const API_KEY = process.env.CHEVALIER_API_KEY; // falls back to provider env var

// 1) Structured output with Zod, via the agentic() helper.
const Person = z.object({ name: z.string(), age: z.number() });
const extract = agentic({ model: MODEL, apiKey: API_KEY }, (text, rt) =>
  rt.run({ prompt: `Extract a person as JSON from: ${text}`, output: Person, maxTokens: 1024 }),
);
const person = await extract("Alice is 30 years old");
console.log("structured:", person.value);

// 2) A tool + manual dispatch.
const rt = new Runtime({ model: MODEL, apiKey: API_KEY });
await rt.tool({
  name: "get_weather",
  description: "Current weather for a city",
  schema: z.object({ city: z.string() }),
  handler: async ({ city }) => `Sunny, 22°C in ${city}`,
});
console.log("tool:", await rt.executeToolCall("get_weather", { city: "Tokyo" }));

// 3) Streaming.
process.stdout.write("stream: ");
for await (const ev of rt.runStream({ prompt: "Say hello in five words.", maxTokens: 256 })) {
  if (ev.type === "content") process.stdout.write(ev.text);
}
process.stdout.write("\n");
process.exit(0);
