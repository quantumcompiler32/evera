// EVERA's brain: OpenAI Chat Completions with tool calling, plus an offline keyword fallback.
import { TOOLS } from './actions.js';

const MODE_IDS = [...new Set(TOOLS.flatMap((t) => t.modes.map((m) => m.id)))];

const catalog = () => TOOLS.map((t) => `- ${t.id}: ${t.name}${t.hot ? ' (hot tool)' : ''}; modes: ${t.modes.map((m) => `${m.id} = "${m.label}"`).join(', ')}`).join('\n');

const SYSTEM = (ctx) => `You are EVERA, the AI that runs a robotic workbench: an SO-101 robot arm next to a maker's bench. Personality: calm, quick, dry British wit, like a brilliant butler-engineer. Address the user as "sir" only occasionally.

Tools on the bench (tool_id: name; modes):
${catalog()}

Rules:
- When the user wants a tool, or an action the arm can do, call run_action IMMEDIATELY. Never ask for confirmation.
- "Hand it to me", "pass me", "grab", or "bring" means mode "pick". Heat-set inserts mean tool "iron", mode "heat". "Hold this steady" means tool "hand", mode "hold". "Clean up" or "put everything back" means tool "dock", mode "dock".
- "Stop", "halt", "freeze" or "wait": call stop_arm.
- "Got it", "thanks, I have it": call got_it.
- Put what you say out loud in the tool's "say" field: one short spoken sentence (max 14 words), natural and witty.
- Without a tool call, answer bench and electronics questions in 1-2 short spoken sentences. No markdown, lists or emoji, because everything you write is read aloud.
- If the request is ambiguous, ask one very short question.

Arm right now: ${ctx}`;

const TOOLS_SPEC = [
  { type: 'function', function: { name: 'run_action', description: 'Make the robot arm perform an action with a bench tool.', parameters: { type: 'object', properties: { tool_id: { type: 'string', enum: TOOLS.map((t) => t.id) }, mode_id: { type: 'string', enum: MODE_IDS }, say: { type: 'string', description: 'Short sentence EVERA speaks aloud.' } }, required: ['tool_id', 'mode_id', 'say'] } } },
  { type: 'function', function: { name: 'stop_arm', description: 'Emergency-stop the arm immediately.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
  { type: 'function', function: { name: 'got_it', description: 'The user has taken the tool that was handed over.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
];

export class Brain {
  constructor(getSettings) {
    this.getSettings = getSettings;
    this.history = [];
  }

  // Returns {say, calls: [{name, args}]}
  async think(text, ctx) {
    const s = this.getSettings();
    if (!s.openaiKey) return this.fallback(text);
    const messages = [{ role: 'system', content: SYSTEM(ctx) }, ...this.history.slice(-12), { role: 'user', content: text }];
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${s.openaiKey}` },
      body: JSON.stringify({ model: s.openaiModel || 'gpt-4.1-mini', messages, tools: TOOLS_SPEC, tool_choice: 'auto' }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err?.error?.message || `OpenAI error ${r.status}`);
    }
    const msg = (await r.json()).choices[0].message;
    const calls = (msg.tool_calls || []).map((c) => {
      let args = {};
      try { args = JSON.parse(c.function.arguments || '{}'); } catch { /* ignore */ }
      return { name: c.function.name, args };
    });
    const say = (msg.content || calls.find((c) => c.args.say)?.args.say || '').trim();
    this.remember(text, say, calls);
    return { say, calls };
  }

  remember(user, say, calls) {
    const did = calls.map((c) => (c.name === 'run_action' ? `[ran ${c.args.tool_id}/${c.args.mode_id}]` : `[${c.name}]`)).join(' ');
    this.history.push({ role: 'user', content: user }, { role: 'assistant', content: `${say} ${did}`.trim() });
    this.history = this.history.slice(-12);
  }

  // Offline: keyword matching, so voice control still works with no API key.
  fallback(text) {
    const t = text.toLowerCase();
    if (/\b(stop|halt|freeze|abort)\b/.test(t)) return { say: 'Stopping.', calls: [{ name: 'stop_arm', args: {} }] };
    if (/\b(got it|have it|thanks|thank you)\b/.test(t)) return { say: 'Very good.', calls: [{ name: 'got_it', args: {} }] };
    const syn = {
      iron: ['solder', 'iron', 'pinecil'], driver: ['screwdriver', 'screw driver', 'driver'], dmm: ['multimeter', 'meter', 'dmm'],
      tape: ['tape', 'measuring'], allen: ['allen', 'hex key', 'alan'], glue: ['glue'], pliers: ['plier'], tweezers: ['tweezer'],
      cutters: ['cutter', 'snips', 'flush'], calipers: ['caliper'], hand: ['helping', 'hold', 'steady'], dock: ['return', 'clean', 'put everything', 'put it all'],
    };
    for (const tool of TOOLS) {
      if ((syn[tool.id] || []).some((w) => t.includes(w))) {
        const mode = tool.id === 'iron' && /(heat|insert)/.test(t) ? tool.modes[1] : tool.modes[0];
        const say = mode.id === 'pick' ? `Fetching the ${tool.name.toLowerCase()}.` : `${mode.label}. On it.`;
        return { say, calls: [{ name: 'run_action', args: { tool_id: tool.id, mode_id: mode.id } }] };
      }
    }
    return { say: "I didn't catch a tool there. Add an OpenAI key in settings and I'll be far cleverer.", calls: [] };
  }
}
