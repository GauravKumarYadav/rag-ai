import { useState, useEffect, useRef, useCallback } from 'react'
import { useChatStore } from '@/stores/chat-store'
import { useClientStore } from '@/stores/client-store'
import { useAuthStore } from '@/stores/auth-store'
import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'
import { ChatMessage } from '@/components/chat/chat-message'
import { ChatInput } from '@/components/chat/chat-input'
import { TypingIndicator } from '@/components/chat/typing-indicator'
import { DocumentsPanel } from '@/components/documents/documents-panel'
import { WebSocketManager, type WsMessage } from '@/lib/ws'
import { MessageSquare } from 'lucide-react'

export function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [docsOpen, setDocsOpen] = useState(false)
  const { token } = useAuthStore()
  const { currentClientId, loadClients } = useClientStore()
  const {
    messages,
    currentConversationId,
    isStreaming,
    streamingContent,
    isWaitingForResponse,
    addMessage,
    setStreaming,
    setStreamingContent,
    appendStreamingContent,
    setWaitingForResponse,
    createConversation,
    loadConversations,
  } = useChatStore()

  const wsRef = useRef<WebSocketManager | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  useEffect(() => {
    loadClients()
    loadConversations()
  }, [loadClients, loadConversations])

  const connectWs = useCallback(
    (clientId: string) => {
      if (wsRef.current) wsRef.current.disconnect()
      if (!token || !clientId) return

      const ws = new WebSocketManager(
        clientId,
        token,
        (msg: WsMessage) => {
          if (msg.type === 'chunk') {
            appendStreamingContent(msg.content || '')
          } else if (msg.type === 'done') {
            setStreaming(false)
            setWaitingForResponse(false)
            if (msg.response) {
              addMessage({ role: 'assistant', content: msg.response })
            }
            setStreamingContent('')
            loadConversations()
          } else if (msg.type === 'error') {
            setStreaming(false)
            setWaitingForResponse(false)
            addMessage({ role: 'assistant', content: `Error: ${msg.detail || 'Unknown error'}` })
            setStreamingContent('')
          }
        },
        () => {},
      )
      ws.connect()
      wsRef.current = ws
    },
    [token, appendStreamingContent, setStreaming, setWaitingForResponse, addMessage, setStreamingContent, loadConversations],
  )

  useEffect(() => {
    if (currentClientId) {
      connectWs(currentClientId)
    }
    return () => {
      wsRef.current?.disconnect()
    }
  }, [currentClientId, connectWs])


  const handleSend = useCallback(
    async (text: string) => {
      let convId = currentConversationId
      if (!convId) {
        convId = await createConversation()
      }

      addMessage({ role: 'user', content: text })
      setWaitingForResponse(true)
      setStreaming(true)
      setStreamingContent('')

      if (wsRef.current?.isConnected) {
        wsRef.current.send({
          type: 'chat',
          message: text,
          conversation_id: convId,
          client_id: currentClientId,
          top_k: 4,
          include_sources: true,
        })
      } else {
        // REST fallback
        try {
          const resp = await fetch('/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              conversation_id: convId,
              client_id: currentClientId,
              message: text,
              stream: false,
              top_k: 4,
              include_sources: true,
            }),
          })
          if (resp.ok) {
            const data = (await resp.json()) as { response?: string }
            addMessage({ role: 'assistant', content: data.response || '' })
          } else {
            addMessage({ role: 'assistant', content: 'Failed to get response.' })
          }
        } catch {
          addMessage({ role: 'assistant', content: 'Connection error.' })
        } finally {
          setStreaming(false)
          setWaitingForResponse(false)
          setStreamingContent('')
          loadConversations()
        }
      }
    },
    [currentConversationId, currentClientId, token, createConversation, addMessage, setWaitingForResponse, setStreaming, setStreamingContent, loadConversations],
  )

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

      <div className="flex flex-1 flex-col min-w-0">
        <Header
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          onToggleDocuments={() => setDocsOpen(!docsOpen)}
        />

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && !isStreaming ? (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                <MessageSquare className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">How can I help you?</h2>
                <p className="mt-1 text-sm text-muted-foreground">Ask a question about your documents or start a conversation.</p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <ChatMessage key={i} role={msg.role} content={msg.content} />
              ))}
              {isStreaming && streamingContent && (
                <ChatMessage role="assistant" content={streamingContent} />
              )}
              {isWaitingForResponse && !streamingContent && <TypingIndicator />}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSend={handleSend} disabled={isWaitingForResponse} />
      </div>

      <DocumentsPanel isOpen={docsOpen} onClose={() => setDocsOpen(false)} />
    </div>
  )
}
