import { getApiBase } from './utils'

export interface WsMessage {
  type: 'chunk' | 'done' | 'error' | 'pong'
  content?: string
  response?: string
  sources?: Source[]
  detail?: string
}

export interface Source {
  id: string
  content: string
  metadata: Record<string, unknown>
  distance?: number
}

type MessageHandler = (msg: WsMessage) => void

export class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private clientId: string
  private token: string
  private onMessage: MessageHandler
  private onStatusChange: (connected: boolean) => void

  constructor(
    clientId: string,
    token: string,
    onMessage: MessageHandler,
    onStatusChange: (connected: boolean) => void,
  ) {
    this.clientId = clientId
    this.token = token
    this.onMessage = onMessage
    this.onStatusChange = onStatusChange
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    const base = getApiBase() || window.location.origin
    const wsBase = base.replace(/^http/, 'ws')
    const url = `${wsBase}/chat/ws/${this.clientId}?token=${this.token}`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this.reconnectAttempts = 0
      this.onStatusChange(true)
    }

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WsMessage
        this.onMessage(data)
      } catch {
        console.error('Failed to parse WebSocket message')
      }
    }

    this.ws.onclose = () => {
      this.onStatusChange(false)
      this.attemptReconnect()
    }

    this.ws.onerror = () => {
      this.onStatusChange(false)
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    setTimeout(() => {
      this.connect()
    }, delay)
  }

  send(data: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
      this.ws = null
    }
  }

  updateClientId(clientId: string) {
    this.clientId = clientId
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.disconnect()
      this.connect()
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}
