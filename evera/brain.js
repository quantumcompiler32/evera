// EVERA's brain: a general conversational AI (Groq / xAI / OpenAI chat completions) with tool calling
// for the arm, web research, and an offline keyword fallback.
import { TOOLS } from './actions.js';

export const RELEASE = /\b(got it|i got it|i've got it|i have it|drop it|drop that|drop|let go|let it go|release|you can let go|open the gripper|thanks i have it)\b/;

// Offline answers to common bench questions; "yes" afterwards fetches the suggested tool.
const RECIPES = {
  pcb: { match: /(pcb|circuit board|solder|electronics|through.?hole|smd|kit)/, fetch: 'iron', say: "For a PCB you'll want the soldering iron, flush cutters, tweezers, the helping hand and the multimeter for checking it after. Shall I bring the iron first?" },
  inserts: { match: /(heat.?set|insert|3d print|threaded)/, fetch: 'iron', mode: 'heat', say: 'Heat-set inserts need the soldering iron in heat-set mode. Shall I start pressing?' },
  measure: { match: /(measure|dimension|length|width|thickness|diameter|size)/, fetch: 'calipers', say: 'Calipers for anything small and precise, the measuring tape for longer runs. Shall I hand you the calipers?' },
  wiring: { match: /(wire|wiring|cable|strip|crimp|connector)/, fetch: 'cutters', say: 'For wiring: flush cutters, pliers, and the multimeter to check continuity. Want the cutters?' },
  debug: { match: /(voltage|continuity|short|resistance|current|battery|test|debug)/, fetch: 'dmm', say: "That's a job for the multimeter. Shall I bring it over?" },
  assemble: { match: /(assemble|screw|bolt|furniture|mount|build)/, fetch: 'driver', say: 'The screwdriver and the allen key should cover assembly. Screwdriver first?' },
  craft: { match: /(glue|stick|craft|cardboard|foam|attach)/, fetch: 'glue', say: 'The glue gun will do it. Shall I fetch it?' },
};

// Words that count as naming each tool.
const SYN = {
  iron: ['solder', 'iron', 'pinecil'], driver: ['screwdriver', 'screw driver', 'driver'], dmm: ['multimeter', 'multi meter', 'meter', 'dmm'],
  tape: ['measuring tape', 'tape measure', 'tape'], allen: ['allen', 'hex key', 'alan key'], glue: ['glue'], pliers: ['plier'], tweezers: ['tweezer'],
  cutters: ['cutter', 'snips', 'flush'], calipers: ['caliper'], hand: ['helping hand', 'third hand', 'hold it steady', 'hold this'], dock: ['return all', 'clean up', 'put everything', 'put it all', 'tidy'],
};
const names = (t, id) => (SYN[id] || []).some((w) => t.includes(w));
const ANSWER = /^(yes|yeah|yep|yup|sure|ok|okay|please|do it|go ahead|that one|this one|the (first|second|third|last|other) one|first|second|third|both)\b/;

const MODE_IDS = [...new Set(TOOLS.flatMap((t) => t.modes.map((m) => m.id)))];

const catalog = () => TOOLS.map((t) => `${t.id} = ${t.name}${t.hot ? ' (hot)' : ''}${t.modes.length > 1 || t.modes[0].id !== 'pick' ? ` [modes: ${t.modes.map((m) => `${m.id} ${m.label}`).join(', ')}]` : ''}`).join('; ');

const SYSTEM = (ctx) => `You are EVERA, the AI that runs a robotic workbench: an SO-101 robot arm that fetches tools for a maker. You're like JARVIS: warm, quick, dry British wit. Address the user as "sir" only occasionally.

How you talk:
- Every reply is SHORT and simple: one or two sentences, under 30 words. It's spoken aloud: no markdown, lists, emoji or URLs.
- Stay focused on the bench and its tools. Everything you say should connect to the tools or the build in front of the user.
- When the user wants to build, fix or make something ("let's build this PCB", "I need to wire this up"): name the few tools it needs, bench tools first (you may mention one or two non-bench parts like an Arduino board or solder), then ask which one they'd like first. Example: "For that PCB we'll want the soldering iron, tweezers and flush cutters, plus your Arduino board. Which shall I bring first?"
- If you ask which tool they want, do NOT call run_action in that same reply; wait for their answer.
- Follow-ups: when they pick ("the iron", "the first one", "yes"), call run_action for it. Keep track of what you suggested.
- Off-topic questions: answer in one short sentence at most, then steer back to the bench.
- Greetings: one friendly sentence, ask what they're building.
- Only call research if the user explicitly asks you to look something up or search. Even then, keep the answer short and tied to the tools.

Bench tools (tool_id: name; modes):
${catalog()}

Arm rules:
- When the user names a specific tool ("pass me the pliers"), call run_action IMMEDIATELY with a short witty "say" (max 12 words). Never ask for confirmation.
- When they describe a need or task instead ("I need to measure this", "let's solder"), suggest the fitting tools and ask which one. Only fetch once they choose.
- "Hand it to me", "pass me", "grab", "bring" mean mode "pick". Heat-set inserts mean tool "iron", mode "heat". "Hold this steady" means tool "hand", mode "hold". "Clean up" or "put everything back" means tool "dock", mode "dock".
- "Stop", "halt", "freeze": call stop_arm. "Got it", "drop it", "let go", "I have it": call got_it (opens the gripper, arm returns to start).
- "Teleop", "let me drive it", "manual control": call start_teleop. "End teleop": call stop_teleop.

Arm right now: ${ctx}`;

const RESEARCH_SYSTEM = `You are EVERA's research module. Search the web, then answer a maker at their workbench in at most two short spoken sentences (under 40 words): plain words, no markdown, lists, URLs or citations. Focus on which tools and parts they need, and end by asking which tool they'd like first. Bench tools: ${TOOLS.map((t) => t.name.toLowerCase()).join(', ')}.`;

const TOOLS_SPEC = [
  { type: 'function', function: { name: 'run_action', description: 'Make the robot arm perform an action with a bench tool.', parameters: { type: 'object', properties: { tool_id: { type: 'string', enum: TOOLS.map((t) => t.id) }, mode_id: { type: 'string', enum: MODE_IDS }, say: { type: 'string', description: 'Short sentence EVERA speaks aloud.' } }, required: ['tool_id', 'mode_id', 'say'] } } },
  { type: 'function', function: { name: 'research', description: 'Search the web. Only when the user explicitly asks to look something up or search.', parameters: { type: 'object', properties: { query: { type: 'string', description: 'What to research, specific and self-contained.' }, say: { type: 'string', description: 'Very short line spoken while searching, e.g. "Let me look into that."' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'stop_arm', description: 'Emergency-stop the arm immediately.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
  { type: 'function', function: { name: 'got_it', description: 'The user has taken the tool that was handed over: open the gripper and return to start.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
  { type: 'function', function: { name: 'start_teleop', description: 'Start teleoperation: the human moves the leader arm and the robot follower copies it.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
  { type: 'function', function: { name: 'stop_teleop', description: 'End teleoperation.', parameters: { type: 'object', properties: { say: { type: 'string' } }, required: ['say'] } } },
];

// Groq, xAI and OpenAI all speak the OpenAI Chat Completions API.
const PROVIDERS = {
  groq: { url: 'https://api.groq.com/openai/v1/chat/completions', model: 'openai/gpt-oss-120b', fallback: 'openai/gpt-oss-20b', name: 'Groq', extra: { reasoning_effort: 'low' } },
  xai: { url: 'https://api.x.ai/v1/chat/completions', model: 'grok-4.7', name: 'xAI' },
  openai: { url: 'https://api.openai.com/v1/chat/completions', model: 'gpt-4.1-mini', name: 'OpenAI' },
};

// Text that will be spoken: drop citation marks, markdown and list syntax.
export function speakable(t) {
  return (t || '')
    .replace(/【[^】]*】/g, '')
    .replace(/\[(?:ran|research|arm|stop_arm|got_it|start_teleop|stop_teleop)[^\]]*\]/gi, '')
    .replace(/\[(\d+|[^\]]*†[^\]]*)\]/g, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/[*_`#>]+/g, '')
    .replace(/^\s*(?:[-•]|\d+[.)])\s+/gm, '')
    .replace(/\s*\n+\s*/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

export class Brain {
  constructor(getSettings) {
    this.getSettings = getSettings;
    this.history = [];
  }

  get provider() {
    return PROVIDERS[this.getSettings().llmProvider] || PROVIDERS.openai;
  }

  async chat(body, retry = true) {
    const s = this.getSettings();
    const p = this.provider;
    const r = await fetch(p.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${s.openaiKey}` },
      body: JSON.stringify({ model: s.openaiModel || p.model, ...(p.extra || {}), ...body }),
    });
    // Groq's free tier limits tokens per minute per model: retry once on the smaller model.
    if (r.status === 429 && retry && p.fallback) return this.chat({ ...body, model: p.fallback }, false);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err?.error?.message || err?.error || `${p.name} error ${r.status}`);
    }
    return (await r.json()).choices[0].message;
  }

  // Returns {say, calls: [{name, args}]}
  async think(text, ctx) {
    const s = this.getSettings();
    if (!s.openaiKey) return this.fallback(text);
    const messages = [{ role: 'system', content: SYSTEM(ctx) }, ...this.history.slice(-10), { role: 'user', content: text }];
    const msg = await this.chat({ messages, tools: TOOLS_SPEC, tool_choice: 'auto' });
    const calls = (msg.tool_calls || []).map((c) => {
      let args = {};
      try { args = JSON.parse(c.function.arguments || '{}'); } catch { /* ignore */ }
      return { name: c.function.name, args };
    });
    let say = speakable(msg.content || calls.find((c) => c.args.say)?.args.say || '');
    let acts = calls;
    // EVERA only fetches when the user names the tool, or answers EVERA's own question
    // ("yes", "the first one"). Otherwise: suggest tools and ask (one more pass with a note).
    const t = text.trim().toLowerCase();
    const lastAsked = [...this.history].reverse().find((m) => m.role === 'assistant')?.content?.includes('?');
    const allowed = (c) => c.name !== 'run_action' || names(t, c.args.tool_id) || (lastAsked && (ANSWER.test(t) || names(t, c.args.tool_id)));
    if (!acts.every(allowed)) {
      const note = { role: 'system', content: 'The user has not chosen a specific tool yet. Do NOT call run_action. In one short sentence, suggest the fitting bench tools for what they said and ask which one they would like.' };
      let again = null;
      try { again = await this.chat({ messages: [...messages, note] }); } catch { /* fall back below */ } // no tools: answer in words
      say = speakable(again?.content).replace(/\w*_?action\s*\(.*$/i, '').trim();
      const q = say.indexOf('?');
      if (q > 0) say = say.slice(0, q + 1); // it's a question: stop there
      say ||= 'Which tool would you like me to bring?';
      acts = calls.filter((c) => c.name !== 'run_action');
    }
    this.remember(text, say, acts);
    return { say, calls: acts };
  }

  // Web research (Groq's built-in browser search). Returns a spoken-style answer.
  async research(query) {
    const p = this.provider;
    const messages = [{ role: 'system', content: RESEARCH_SYSTEM }, ...this.history.slice(-6), { role: 'user', content: query }];
    let msg;
    if (this.getSettings().llmProvider === 'groq') {
      msg = await this.chat({ model: 'openai/gpt-oss-120b', messages, tools: [{ type: 'browser_search' }], tool_choice: 'required', reasoning_effort: 'medium' });
    } else {
      msg = await this.chat({ messages }); // no web search on this provider: answer from knowledge
    }
    const answer = speakable(msg.content) || "I couldn't find a solid answer on that one.";
    this.history.push({ role: 'system', content: `Research result for "${query}": ${answer}` });
    this.history = this.history.slice(-10);
    return answer;
  }

  remember(user, say, calls) {
    this.history.push({ role: 'user', content: user }, { role: 'assistant', content: say || '(acted)' });
    for (const c of calls) {
      const what = c.name === 'run_action' ? `fetched the ${c.args.tool_id} (${c.args.mode_id})` : c.name.replace('_', ' ');
      this.history.push({ role: 'system', content: `Arm action done: ${what}.` });
    }
    this.history = this.history.slice(-10);
  }

  // Offline brain (no key, or the API is down): keyword matching plus a few project "recipes",
  // so it can still answer "what do I need for a PCB?" and fetch on "yes".
  fallback(text) {
    const t = text.toLowerCase();
    const run = (tool, mode, say) => ({ say, calls: [{ name: 'run_action', args: { tool_id: tool, mode_id: mode } }] });
    if (/\b(stop|halt|freeze|abort)\b/.test(t)) return { say: 'Stopping.', calls: [{ name: 'stop_arm', args: {} }] };
    if (RELEASE.test(t)) return { say: 'Releasing it now.', calls: [{ name: 'got_it', args: {} }] };
    if (/\b(end|stop|exit|finish) tele ?op/.test(t)) return { say: 'Ending teleop.', calls: [{ name: 'stop_teleop', args: {} }] };
    if (/tele ?op|manual control|let me drive|drive it myself/.test(t)) return { say: 'Teleop engaged. The arm is yours.', calls: [{ name: 'start_teleop', args: {} }] };
    if (/^(hi|hello|hey|good (morning|afternoon|evening)|yo)\b/.test(t)) return { say: 'Hello. Bench is online and ready when you are.', calls: [] };
    if (this.pending && /^(yes|yeah|yep|sure|ok|okay|please|do it|go ahead|go for it)\b/.test(t)) {
      const p = this.pending;
      this.pending = null;
      return run(p.tool, p.mode, `Fetching the ${TOOLS.find((x) => x.id === p.tool).name.toLowerCase()}.`);
    }
    const syn = SYN;
    const asking = /\b(what|which|how|why|should|recommend|need|suggest)\b/.test(t) || t.includes('?');
    for (const [key, r] of Object.entries(RECIPES)) {
      if (asking && r.match.test(t)) {
        this.pending = { tool: r.fetch, mode: r.mode || 'pick' };
        return { say: r.say, calls: [] };
      }
    }
    for (const tool of TOOLS) {
      if ((syn[tool.id] || []).some((w) => t.includes(w))) {
        const mode = tool.id === 'iron' && /(heat|insert)/.test(t) ? tool.modes[1] : tool.modes[0];
        const say = mode.id === 'pick' ? `Fetching the ${tool.name.toLowerCase()}.` : `${mode.label}. On it.`;
        return run(tool.id, mode.id, say);
      }
    }
    if (/(heat.?set|insert)/.test(t)) return run('iron', 'heat', 'Heat-set mode. Fingers clear, please.');
    if (asking) return { say: "My full brain is offline at the moment, sir. I can still fetch any tool by name.", calls: [] };
    return { say: "I didn't catch a tool there. Try naming it, like the multimeter.", calls: [] };
  }
}
