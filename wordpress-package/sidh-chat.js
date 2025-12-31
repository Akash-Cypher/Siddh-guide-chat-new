// The URL to your AWS backend.
// Replace this with the actual API Gateway URL.
const API_URL = 'https://YOUR_AWS_API_GATEWAY_URL/chat';

document.addEventListener('DOMContentLoaded', () => {
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

    chatToggleButton.addEventListener('click', () => {
        const isVisible = chatWindow.classList.toggle('visible');
        if (isVisible) {
            stopProactiveMessaging();
            userInput.focus();
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

    async function fetchBotResponse(userMessage) {
        const botMessageElement = addMessage('', 'bot');
        botMessageElement.classList.add('typing');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            botMessageElement.classList.remove('typing');
            typewriterEffect(botMessageElement, data.answer);

        } catch (error) {
            console.error('Error fetching bot response:', error);
            botMessageElement.classList.remove('typing');
            typewriterEffect(botMessageElement, "Sorry, something went wrong. Please try again.");
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
