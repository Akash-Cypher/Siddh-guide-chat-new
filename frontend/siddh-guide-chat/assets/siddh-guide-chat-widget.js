document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = '/wp-json/siddh/v1';
  const STORAGE_KEY = 'siddh_session_id';

  let SESSION_ID = localStorage.getItem(STORAGE_KEY);

  if (!SESSION_ID) {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      SESSION_ID = window.crypto.randomUUID();
    } else {
      SESSION_ID = 'sess_' + Math.random().toString(36).slice(2) + Date.now();
    }
    localStorage.setItem(STORAGE_KEY, SESSION_ID);
  }

  const chatToggleButton = document.getElementById('chatToggleButton');
  const chatWindow = document.getElementById('chatWindow');
  const proactiveMessage = document.getElementById('proactiveMessage');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatMessages = document.getElementById('chatMessages');

  if (!chatToggleButton || !chatWindow || !proactiveMessage || !chatForm || !userInput || !chatMessages) {
    return;
  }

  const proactiveMessages = [
    'Hi! Do you need any guidance?',
    'Hello! Are you looking for any course-related help?',
    'Need help with our services? Ask me!',
    'Curious about our courses? I can help.'
  ];

  let proactiveIndex = 0;
  let proactiveInterval = null;
  let historyLoaded = false;

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
    if (proactiveInterval) {
      clearInterval(proactiveInterval);
      proactiveInterval = null;
    }
    proactiveMessage.classList.remove('visible');
  }

  async function loadHistory() {
    try {
      const response = await fetch(`${API_BASE}/history/${encodeURIComponent(SESSION_ID)}`, {
        method: 'GET',
        headers: {
          Accept: 'application/json'
        }
      });

      if (!response.ok) {
        return;
      }

      const data = await response.json();
      const msgs = Array.isArray(data.messages) ? data.messages : [];

      clearMessages();

      for (const m of msgs) {
        const role = (m.role || '').toLowerCase();
        const text = (m.text || '').trim();
        if (!text) continue;

        if (role === 'user') {
          addMessage(text, 'user');
        } else if (role === 'assistant') {
          addMessage(text, 'bot');
        }
      }
    } catch (err) {
      console.warn('History load failed:', err);
    }
  }

  function typewriterEffect(element, text) {
    let i = 0;
    element.textContent = '';
    const speed = 20;

    function type() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i += 1;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setTimeout(type, speed);
      }
    }

    type();
  }

  async function fetchBotResponse(userMessage) {
    const botMessageElement = addMessage('...', 'bot');
    botMessageElement.classList.add('typing');

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: SESSION_ID
        })
      });

      if (!response.ok) {
        let errText = `HTTP error: ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.detail) {
            errText = errJson.detail;
          }
        } catch (e) {
          // ignore parse issue
        }
        throw new Error(errText);
      }

      const data = await response.json();
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = '';
      typewriterEffect(botMessageElement, data.answer || '');
    } catch (error) {
      console.error('Error fetching bot response:', error);
      botMessageElement.classList.remove('typing');
      botMessageElement.textContent = 'Sorry, something went wrong. Please try again.';
    }
  }

  chatToggleButton.addEventListener('click', async () => {
    const isVisible = chatWindow.classList.toggle('visible');

    if (isVisible) {
      stopProactiveMessaging();
      userInput.focus();

      if (!historyLoaded) {
        await loadHistory();
        historyLoaded = true;
      }
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

  setTimeout(startProactiveMessaging, 3000);
});