'use strict';
const $ = selector => document.querySelector(selector);
const wrap = $('#wrap'), feed = $('#feed'), input = $('#q');
let conversationId = localStorage.getItem('activeConversationId'), busy = false, navigation = 0;
const welcome = wrap.innerHTML;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function scroll() { feed.scrollTop = feed.scrollHeight; }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { const data = await response.json(); detail = typeof data.detail === 'string' ? data.detail : detail; } catch {}
    throw new Error(`${detail} (HTTP ${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}
const json = (method, body) => ({method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
function remember(id) { conversationId = id ? String(id) : null; if (conversationId) localStorage.setItem('activeConversationId', conversationId); else localStorage.removeItem('activeConversationId'); }
function action(label, handler, solid = false) { const b = document.createElement('button'); b.className = 'chip' + (solid ? ' solid' : ''); b.textContent = label; b.onclick = handler; return b; }
function sources(card, items) {
  if (!items?.length) return;
  const details = document.createElement('details'); details.className = 'sources';
  const summary = document.createElement('summary'); summary.textContent = `Sources (${items.length})`; details.append(summary);
  for (const s of items) {
    const source = document.createElement('div'); source.className = 'source';
    source.innerHTML = `<strong>${esc(s.filename || s.file_name || s.document || 'Source document')}</strong><div>Page ${esc(s.page ?? '')}${s.section ? ' · '+esc(s.section) : ''}</div><p>${esc(s.excerpt || '')}</p>`;
    details.append(source);
  }
  card.append(details);
}
function message(role, text, citations = []) {
  const row = document.createElement('article'); row.className = 'row ' + (role === 'user' ? 'me' : 'ai');
  row.innerHTML = `<div class="av">${role === 'user' ? 'You' : 'DD'}</div><div class="card"><div class="who">${role === 'user' ? 'You' : 'DrillDoc Assistant'}</div><div class="answer" dir="auto"></div></div>`;
  const body = row.querySelector('.answer'), card = row.querySelector('.card'); body.textContent = text;
  if (role !== 'user') {
    const acts = document.createElement('div'); acts.className = 'acts';
    acts.append(action('Copy answer', async e => { try { await navigator.clipboard.writeText(body.textContent); e.target.textContent = 'Copied'; } catch { e.target.textContent = 'Copy unavailable'; } })); card.append(acts);
    sources(card, citations);
  }
  wrap.append(row); scroll(); return {row, card, body};
}
function closeMenu() { $('#sidebar').classList.remove('open'); $('#menu').setAttribute('aria-expanded','false'); }
function setBusy(value) { busy = value; $('#send').disabled = value; input.disabled = value; $('#newChat').disabled = value; document.querySelectorAll('[data-q], #history button').forEach(b => b.disabled = value); }
function clearChat() { if (busy) return; ++navigation; remember(null); wrap.innerHTML = welcome; closeMenu(); loadHistory(); input.focus(); }
async function loadHistory() {
  try {
    const rows = await api('/api/conversations'); $('#history-count').textContent = rows.length; $('#history').replaceChildren(); $('#history-error').textContent = '';
    for (const item of rows) {
      const line = document.createElement('div'); line.style.display = 'flex';
      const open = action(item.title || 'New chat', () => openChat(item.id)); open.style.flex = '1'; open.style.minWidth = '0'; if (String(item.id) === conversationId) open.classList.add('on');
      const rename = action('✎', async () => { const title = prompt('Conversation title', item.title); if (title?.trim()) { try { await api('/api/conversations/'+encodeURIComponent(item.id), json('PATCH',{title:title.trim()})); loadHistory(); } catch(e) { $('#history-error').textContent=e.message; } } }); rename.setAttribute('aria-label','Rename conversation');
      const del = action('×', async () => { if (!confirm('Delete this conversation?')) return; try { await api('/api/conversations/'+encodeURIComponent(item.id), {method:'DELETE'}); if (String(item.id) === conversationId) clearChat(); else loadHistory(); } catch(e) { $('#history-error').textContent=e.message; } }); del.setAttribute('aria-label','Delete conversation');
      line.append(open, rename, del); $('#history').append(line);
    }
    if (!rows.length) $('#history-error').textContent = 'No conversations yet.';
    setBusy(busy);
  } catch(e) { $('#history-error').textContent = 'History unavailable: '+e.message; }
}
async function openChat(id) {
  if (busy) return; const token = ++navigation;
  try {
    const data = await api('/api/v1/conversations/'+encodeURIComponent(id)); if (token !== navigation) return;
    remember(id); wrap.replaceChildren(); for (const m of data.messages || []) message(m.role,m.content,m.sources || []);
    if (!data.messages?.length) wrap.innerHTML = welcome; closeMenu(); loadHistory();
  } catch(e) { $('#history-error').textContent = e.message; }
}
async function checkStatus() {
  const results = await Promise.allSettled([api('/health'), api('/ready')]);
  const ready = results[1].status === 'fulfilled'; $('#service-status').textContent = ready ? 'RAG ready' : 'RAG unavailable'; $('#status-dot').parentElement.classList.toggle('unavailable', !ready);
  $('#diagnostics').textContent = results.map((r,i) => `${i ? 'Readiness' : 'Health'}: ${r.status === 'fulfilled' ? JSON.stringify(r.value,null,2) : r.reason.message}`).join('\n\n');
}
async function ask(text) {
  if (busy || !text.trim()) return;
  text = text.trim(); ++navigation; setBusy(true); input.value = ''; if (wrap.innerHTML === welcome) wrap.replaceChildren();
  message('user',text); const live = message('assistant','');
  live.body.innerHTML = '<div class="typing" aria-label="Waiting for answer"><i></i><i></i><i></i></div>';
  let answer = '', citations = [], completed = false;
  try {
    const response = await fetch('/api/v1/chat/stream', json('POST',{query:text, conversation_id:conversationId, language:/[\u0600-\u06ff]/.test(text)?'fa':'en'}));
    if (!response.ok) { let detail = 'Chat request failed'; try { const d = await response.json(); if (typeof d.detail === 'string') detail=d.detail; } catch {} throw new Error(`${detail} (HTTP ${response.status})`); }
    if (!response.body) throw new Error('Streaming response unavailable');
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
    function event(block) {
      const kind = block.match(/^event:\s*(.+)$/m)?.[1]?.trim();
      const payload = block.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n'); if (!payload) return;
      const data = JSON.parse(payload);
      if (kind === 'start') { remember(data.conversation_id); }
      if (kind === 'answer_delta') { answer += data.text || ''; live.body.textContent = answer; scroll(); }
      if (kind === 'citation') citations.push(data);
      if (kind === 'error') throw new Error(data.detail || data.message || 'Stream failed');
      if (kind === 'done') { completed = true; remember(data.conversation_id); if (!answer) live.body.textContent = data.answer || ''; if (!citations.length) citations = data.sources || []; }
    }
    while (true) { const chunk = await reader.read(); buffer += decoder.decode(chunk.value || new Uint8Array(), {stream:!chunk.done}); buffer = buffer.replace(/\r\n/g,'\n'); let boundary; while ((boundary=buffer.indexOf('\n\n')) >= 0) { event(buffer.slice(0,boundary)); buffer=buffer.slice(boundary+2); } if (chunk.done) break; }
    if (buffer.trim()) event(buffer);
    if (!completed) throw new Error('Response stream ended before completion');
    sources(live.card,citations);
  } catch(e) {
    live.card.classList.add('err'); live.body.textContent = answer ? answer+'\n\nResponse interrupted: '+e.message : 'No answer received. '+e.message;
    const acts = live.card.querySelector('.acts'); acts.append(action('Try again', () => { if (!busy) ask(text); },true),action('Check service',()=>{ checkStatus(); $('#settings-dialog').showModal(); }));
  } finally { setBusy(false); loadHistory(); scroll(); input.focus(); }
}
$('#send').onclick = () => ask(input.value);
input.addEventListener('keydown',e=>{ if(e.key==='Enter' && !e.isComposing) { e.preventDefault(); ask(input.value); } });
document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>ask(b.dataset.q));
$('#newChat').onclick=clearChat;
$('#menu').onclick=()=>{ const open=$('#sidebar').classList.toggle('open'); $('#menu').setAttribute('aria-expanded',String(open)); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeMenu(); });
$('#settings').onclick=()=>{ checkStatus(); $('#settings-dialog').showModal(); };
$('#close-settings').onclick=()=>$('#settings-dialog').close(); $('#refresh-status').onclick=checkStatus;
async function checkStatus() {
  try {
    const response = await fetch('/ready');
    const data = await response.json();
    $('#service-status').textContent = response.ok ? 'RAG ready' : 'RAG unavailable';
    $('#status-dot').parentElement.classList.toggle('unavailable', !response.ok);
    $('#diagnostics').textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    $('#service-status').textContent = 'Server unavailable';
    $('#status-dot').parentElement.classList.add('unavailable');
    $('#diagnostics').textContent = e.message;
  }
}
loadHistory(); checkStatus(); if(conversationId) openChat(conversationId);
