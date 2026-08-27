export class VlaBridgeClient {
  constructor(url, token) {
    this.url = url;
    this.token = token;
    this.socket = null;
    this.pending = new Map();
  }

  async connect() {
    if (this.socket?.readyState === WebSocket.OPEN) return;

    await new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      this.socket = socket;
      let handshakeFinished = false;

      const fail = (error) => {
        if (handshakeFinished) return;
        handshakeFinished = true;
        reject(error instanceof Error ? error : new Error("WebSocket connection failed"));
      };
      socket.addEventListener("error", fail, { once: true });
      socket.addEventListener("open", () => {
        socket.send(JSON.stringify({
          type: "hello",
          protocol: "vla-bridge.v1",
          token: this.token,
          client: "simulation-web",
        }));
      }, { once: true });
      socket.addEventListener("message", (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch (error) {
          fail(new Error("Server returned invalid JSON"));
          socket.close(1002, "invalid JSON");
          return;
        }
        if (message.type === "ready") {
          if (handshakeFinished) return;
          handshakeFinished = true;
          socket.removeEventListener("error", fail);
          resolve();
          return;
        }
        if (message.type === "error" && !handshakeFinished) {
          fail(new Error(`${message.code}: ${message.message}`));
          socket.close(1008, "handshake rejected");
          return;
        }
        if (message.type === "action" || message.type === "error") {
          const waiter = this.pending.get(message.request_id);
          if (!waiter) return;
          this.pending.delete(message.request_id);
          if (message.type === "action") waiter.resolve(message);
          else waiter.reject(new Error(`${message.code}: ${message.message}`));
        }
      });
      socket.addEventListener("close", () => {
        fail(new Error("WebSocket closed during handshake"));
        for (const waiter of this.pending.values()) waiter.reject(new Error("WebSocket closed"));
        this.pending.clear();
      });
    });
  }

  async getAction(state, {
    instruction = null,
    images = null,
    episodeId = null,
    step = null,
    timeoutMs = 300000,
    returnResponse = false,
  } = {}) {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error("VLA bridge is not connected");
    const requestId = crypto.randomUUID();
    const result = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error("Action request timed out"));
      }, timeoutMs);
      this.pending.set(requestId, {
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
    });
    this.socket.send(JSON.stringify({
      type: "state",
      request_id: requestId,
      episode_id: episodeId,
      step,
      timestamp_ms: Date.now(),
      instruction,
      state,
      images,
    }));
    const response = await result;
    return returnResponse ? response : response.action;
  }

  close() {
    this.socket?.close(1000, "client shutdown");
  }
}
