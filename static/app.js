let sessionId = null;

const $chat = document.getElementById("chat");
const $form = document.getElementById("form");
const $input = document.getElementById("input");
const $clear = document.getElementById("clear");

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="role">${role === "user" ? "You" : "Bot"}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  $chat.appendChild(wrap);
  $chat.scrollTop = $chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[c]));
}

async function sendMessage(message) {
  addMessage("user", message);
  $input.value = "";
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    const json = await res.json();
    sessionId = json.session_id;
    addMessage("bot", json.reply);
  } catch {
    addMessage("bot", "Network error. Is the server running?");
  }
}

$form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $input.value.trim();
  if (!text) return;
  sendMessage(text);
});

$clear.addEventListener("click", () => {
  if (!sessionId) return;
  sendMessage("clear history");
});
