const WebSocket = require("wss");

const wss = new WebSocket.Server({ port: 8080 });

console.log("WebSocket server is running on wss://localhost:8080");

let connectionID = 1;

function broadcast(sender, dataString) {
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN && client !== sender) {
      client.send(dataString);
    }
  }
}

wss.on("connection", (ws) => {
  const id = connectionID++;
  console.log("New client connected:", id);

  ws.send("Welcome to the WebSocket server!");
  broadcast(ws, "New user connected: user" + id);

  ws.on("message", (raw) => {
    const text = Buffer.isBuffer(raw) ? raw.toString("utf8") : String(raw);
    const out = `user${id}: ${text}`;
    broadcast(ws, out);
    ws.send(`You: ${text}`);
  });

  ws.on("close", () => {
    console.log("Client disconnected:", id);
    broadcast(ws, "User left: user" + id);
  });

  ws.on("error", (err) => {
    console.error("WS error:", err);
  });
});
