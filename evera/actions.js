// EVERA action config. Edit this file as skills start working on the real arm.
//
// Every mode is sent to the dashboard as POST /api/policy {ckpt, task, secs}:
//   task  = the language prompt (pi0.5 reads it; ACT ignores it)
//   ckpt  = checkpoint folder on the GMKtec ('' = the dashboard's default model)
//   secs  = how long the policy runs before it stops and returns to rest
//   motion = how the 3D arm animates while it runs: handover | press | hold | dock

const ACT_PICK = '/home/aup/hackathon/outputs/rl_act/checkpoints/last/pretrained_model';

const pick = (thing, ckpt = '') => ({
  id: 'pick',
  label: 'Hand over',
  task: `pick up the ${thing} and hand it to me`,
  ckpt,
  secs: 30,
  motion: 'handover',
});

export const TOOLS = [
  { id: 'iron', name: 'Soldering iron', icon: 'iron', hot: true, modes: [
    pick('soldering iron', ACT_PICK),
    { id: 'heat', label: 'Heat-set insert', task: 'press the heat-set insert into the part with the soldering iron', ckpt: '', secs: 45, motion: 'press' },
  ] },
  { id: 'driver', name: 'Screwdriver', icon: 'driver', modes: [pick('screwdriver')] },
  { id: 'dmm', name: 'Multimeter', icon: 'dmm', modes: [pick('multimeter')] },
  { id: 'tape', name: 'Measuring tape', icon: 'tape', modes: [pick('measuring tape')] },
  { id: 'allen', name: 'Allen key', icon: 'allen', modes: [pick('allen key')] },
  { id: 'glue', name: 'Glue gun', icon: 'glue', hot: true, modes: [pick('hot glue gun')] },
  { id: 'pliers', name: 'Pliers', icon: 'pliers', modes: [pick('pliers')] },
  { id: 'tweezers', name: 'Tweezers', icon: 'tweezers', modes: [pick('tweezers')] },
  { id: 'cutters', name: 'Flush cutters', icon: 'cutters', modes: [pick('flush cutters')] },
  { id: 'calipers', name: 'Calipers', icon: 'calipers', modes: [pick('calipers')] },
  { id: 'hand', name: 'Helping hand', icon: 'hand', modes: [
    { id: 'hold', label: 'Hold steady', task: 'hold the part steady', ckpt: '', secs: 60, motion: 'hold' },
  ] },
  { id: 'dock', name: 'Return all', icon: 'dock', modes: [
    { id: 'dock', label: 'Return all', task: 'return all tools to the dock', ckpt: '', secs: 60, motion: 'dock' },
  ] },
];

// Left arc top-to-bottom, then right arc top-to-bottom (matches the layout).
export const LEFT = ['iron', 'driver', 'dmm', 'tape', 'allen', 'glue'];
export const RIGHT = ['pliers', 'tweezers', 'cutters', 'calipers', 'hand', 'dock'];

export const toolById = (id) => TOOLS.find((t) => t.id === id);
