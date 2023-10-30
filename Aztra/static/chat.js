const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");
const API_URL = "/assist";
const table_name = document.querySelector('#tablename').value;

var md = window.markdownit();


sendButton.addEventListener("click", () => {
    handleOutgoingChat();
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleOutgoingChat();
    }
});

const handleOutgoingChat = async () => {
    const userText = chatInput.value.trim();
    if (!userText) return;

    appendUserMessage(userText);
    chatInput.value = "";

    try {
        const response = await getChatResponse(userText);
        appendAssistantMessage(response);
    } catch (error) {
        appendErrorMessage("Oops! Something went wrong while retrieving the response. Please try again.");
    }
};

const getChatResponse = async (userText) => {
    const requestBody = {
        userText: userText,
        table_name: table_name
    };
    const requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
    };

    const response = await fetch(API_URL, requestOptions);
    if (!response.ok) {
        throw new Error("Request to /assist failed.");
    }

    const data = await response.json();
    return data.text;
};

const appendUserMessage = (message) => {
    const userMessage = createMessageElement("outgoing", message);
    chatContainer.appendChild(userMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
};

const appendAssistantMessage = (message) => {
    const assistantMessage = createMessageElement("incoming", message);
    chatContainer.appendChild(assistantMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
};

const appendErrorMessage = (message) => {
    const errorMessage = createMessageElement("error", message);
    chatContainer.appendChild(errorMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
};

const createMessageElement = (className, content) => {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat", className);
    messageDiv.innerHTML = `<div class="chat-content">
        <div class="chat-details">
            <img src="${className === "outgoing" ? "user.jpg" : "https://img1.wsimg.com/isteam/ip/85d08d79-91db-410d-a442-2255f9b47c90/Aztra%20Bot%20Ver%203%20copy.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25"}" alt="${className}-img">
        <div style="flex-direction:column; width:100%; overlay-x: hidden;">${md.render(content)}</div>
        </div>
    </div>`;

    hljs.highlightAll();
    return messageDiv;
};

themeButton.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});

deleteButton.addEventListener("click", () => {
    if (confirm("Are you sure you want to delete all the chats?")) {
        clearChat();
    }
});

const clearChat = () => {
    chatContainer.innerHTML = "";
    localStorage.removeItem("all-chats");
};

// Load saved theme
const themeColor = localStorage.getItem("themeColor");
if (themeColor) {
    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
}

// Load saved chats
const savedChats = localStorage.getItem("all-chats");
if (savedChats) {
    chatContainer.innerHTML = savedChats;
    chatContainer.scrollTop = chatContainer.scrollHeight;
};