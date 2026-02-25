document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------------------
  // Configuration (WordPress Proxy)
  // -------------------------------------------------------------------------
  const API_BASE = "/wp-json/siddh/v1";

  // Persist session across refresh/hard refresh
  const STORAGE_KEY = "siddh_session_id";
  let SESSION_ID = localStorage.getItem(STORAGE_KEY);

  if (!SESSION_ID) {
    if (window.crypto && crypto.randomUUID) {
      SESSION_ID = crypto.randomUUID();
    } else {
      SESSION_ID = "sess_" + Math.random().toString(36).slice(2) + Date.now();
    }
    localStorage.setItem(STORAGE_KEY, SESSION_ID);
  }
  // -------------------------------------------------------------------------

  const chatToggleButton = document.getElementById('chatToggleButton');
  const chatWindow = document.getElementById('chatWindow');
  const proactiveMessage = document.getElementById('proactiveMessage');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatMessages = document.getElementById('chatMessages');

  const proactiveMessages = [
    "Hi! Do you need any guidance?",
    "Hello! Are you looking for any course-related stuff?",
    "Need help with our services? Ask me!",
    "Curious about our courses? I can help."
  ];
  let proactiveIndex = 0;
  let proactiveInterval;

  function showProactiveMessage() {
    proactiveMessage.textContent = proactiveMessages[proactiveIndex];
    proactiveMessage.classList.add('visible');

    setTimeout(() => {
      proactiveMessage.classList.remove('visible');
    }, 5000);

    proactiveIndex = (proactiveIndex + 1) % proactiveMessages.length;
  }

  function startProactiveMessaging() {
    if (chatWindow.classList.contains('visible')) return;
    stopProactiveMessaging();
    proactiveInterval = setInterval(showProactiveMessage, 8000);
  }

  function stopProactiveMessaging() {
    clearInterval(proactiveInterval);
    proactiveMessage.classList.remove('visible');
  }

  chatToggleButton.addEventListener('click', async () => {
    const isVisible = chatWindow.classList.toggle('visible');
    if (isVisible) {
      stopProactiveMessaging();
      userInput.focus();

      // Load history only when opening chat (less noise)
      await loadHistory();
    } else {
      startProactiveMessaging();
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const messageText = userInput.value.trim();
    if (!messageText) return;

    addMessage(messageText, 'user');
    userInput.value = '';
    fetchBotResponse(messageText);
  });

  function addMessage(text, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageElement;
  }

  function clearMessages() {
    chatMessages.innerHTML = '';
  }

  async function loadHistory() {
    try {
      const response = await fetch(`${API_BASE}/history/${encodeURIComponent(SESSION_ID)}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (!response.ok) return;

      const data = await response.json();
      const msgs = data.messages || [];

      // Rebuild UI from stored chat
      clearMessages();
      for (const m of msgs) {
        const role = (m.role || "").toLowerCase();
        const text = (m.text || "").trim();
        if (!text) continue;

        if (role === "user") addMessage(text, "user");
        else addMessage(text, "bot");
      }

    } catch (err) {
      console.warn("History load failed:", err);
    }
  }

  async function fetchBotResponse(userMessage) {
    const botMessageElement = addMessage('...', 'bot');
    botMessageElement.classList.add('typing');

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          session_id: SESSION_ID
        }),
      });

      if (!response.ok) {
        let errText = `HTTP error: ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson?.detail) errText = errJson.detail;
        } catch (_) {}
        throw new Error(errText);
      }

      const data = await response.json();
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = '';
      typewriterEffect(botMessageElement, data.answer || "");

    } catch (error) {
      console.error('Error fetching bot response:', error);
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = "Sorry, something went wrong. Please try again.";
    }
  }

  function typewriterEffect(element, text) {
    let i = 0;
    element.textContent = '';
    const speed = 30;

    function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setTimeout(type, speed);
      }
    }
    type();
  }

  setTimeout(startProactiveMessaging, 3000);
});
