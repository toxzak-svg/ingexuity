const conversation = document.querySelector("#conversation");
const composer = document.querySelector("#composer");
const promptInput = document.querySelector("#prompt");
const responseLength = document.querySelector("#responseLength");
const sendButton = document.querySelector("#sendButton");
const cancelButton = document.querySelector("#cancelButton");
const continueButton = document.querySelector("#continueButton");
const contextNotice = document.querySelector("#contextNotice");
const runtimeStatus = document.querySelector("#runtimeStatus");

const STORAGE_KEY = "ingexuity-hf-cpu-messages-v1";
let messages = loadMessages();
let retainedMessages = [...messages];
let activeController = null;
let lastAssistantText = "";

function loadMessages() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.slice(-12).filter(validMessage) : [];
  } catch {
    return [];
  }
}

function validMessage(message) {
  return message && ["user", "assistant", "system"].includes(message.role) && typeof message.content === "string";
}

function saveMessages() {
  messages = messages.slice(-12);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function addMessage(role, text = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "IngExuity";
  const content = document.createElement("pre");
  content.textContent = text;
  article.append(label, content);
  conversation.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return content;
}

function renderHistory() {
  conversation.replaceChildren();
  for (const message of messages) addMessage(message.role, message.content);
}

function selectedMaxTokens() {
  return Number(responseLength.value);
}

function setGenerating(generating) {
  sendButton.disabled = generating;
  promptInput.disabled = generating;
  responseLength.disabled = generating;
  cancelButton.hidden = !generating;
}

function parseSseRecord(record) {
  let event = "message";
  const data = [];
  for (const line of record.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (!data.length) return null;
  return { event, data: JSON.parse(data.join("\n")) };
}

async function consumeSse(response, onEvent) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "request_failed" }));
    throw new Error(error.message || error.error || `HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const record = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseRecord(record);
      if (parsed) onEvent(parsed.event, parsed.data);
    }
    if (done) break;
  }
}

async function streamRequest(endpoint, payload, assistantNode) {
  activeController = new AbortController();
  setGenerating(true);
  continueButton.hidden = true;
  contextNotice.hidden = true;
  let done = null;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: activeController.signal,
    });
    await consumeSse(response, (event, data) => {
      if (event === "meta") {
        contextNotice.hidden = !data.context_trimmed;
        if (Array.isArray(data.retained_messages)) retainedMessages = data.retained_messages;
      } else if (event === "token") {
        assistantNode.textContent += data.text;
        lastAssistantText = assistantNode.textContent;
      } else if (event === "error") {
        throw new Error(data.message || data.error);
      } else if (event === "done") {
        done = data;
      }
    });
    if (!done) throw new Error("The stream ended without a completion record.");
    continueButton.hidden = !done.can_continue;
    if (done.finish_reason === "length") {
      continueButton.hidden = !done.can_continue;
    } else {
      continueButton.hidden = true;
    }
    return done;
  } finally {
    activeController = null;
    setGenerating(false);
  }
}

async function sendMessage(text) {
  messages.push({ role: "user", content: text });
  saveMessages();
  addMessage("user", text);
  const assistantNode = addMessage("assistant", "");
  lastAssistantText = "";
  try {
    await streamRequest("/api/chat", {
      messages,
      max_new_tokens: selectedMaxTokens(),
      stream: true,
    }, assistantNode);
    if (assistantNode.textContent.trim()) {
      messages.push({ role: "assistant", content: assistantNode.textContent });
      saveMessages();
    }
  } catch (error) {
    if (error.name !== "AbortError") assistantNode.textContent = `Error: ${error.message}`;
  }
}

async function sendContinuation(retained, priorText, maxTokens) {
  const assistantNode = addMessage("assistant", "");
  try {
    await streamRequest("/api/chat/continue", {
      messages: retained,
      prior_text: priorText,
      max_new_tokens: maxTokens,
      stream: true,
    }, assistantNode);
    if (assistantNode.textContent.trim()) {
      messages.push({ role: "assistant", content: assistantNode.textContent });
      saveMessages();
    }
  } catch (error) {
    if (error.name !== "AbortError") assistantNode.textContent = `Error: ${error.message}`;
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text) return;
  promptInput.value = "";
  await sendMessage(text);
});

cancelButton.addEventListener("click", () => activeController?.abort());
continueButton.addEventListener("click", () => sendContinuation(retainedMessages, lastAssistantText, selectedMaxTokens()));

async function refreshRuntime() {
  try {
    const response = await fetch("/api/runtime", { cache: "no-store" });
    const runtime = await response.json();
    runtimeStatus.textContent = runtime.state === "ready"
      ? `Ready · ${runtime.threads} threads · ${runtime.context_tokens.toLocaleString()} tokens`
      : `Runtime: ${runtime.state}`;
    runtimeStatus.dataset.state = runtime.state;
  } catch {
    runtimeStatus.textContent = "Runtime unavailable";
    runtimeStatus.dataset.state = "failed";
  }
}

renderHistory();
refreshRuntime();
setInterval(refreshRuntime, 15000);
