import { create } from 'zustand'
import { api } from '@/lib/api'

export interface Message {
  id?: string
  role: 'user' | 'assistant'
  content: string
}

export interface Conversation {
  id: string
  title: string
  created_at?: string
  updated_at?: string
  messages?: Message[]
}

interface ChatState {
  conversations: Conversation[]
  currentConversationId: string | null
  messages: Message[]
  isStreaming: boolean
  streamingContent: string
  isWaitingForResponse: boolean

  setCurrentConversation: (id: string | null) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  setStreaming: (streaming: boolean) => void
  setStreamingContent: (content: string) => void
  appendStreamingContent: (chunk: string) => void
  setWaitingForResponse: (waiting: boolean) => void

  loadConversations: () => Promise<void>
  createConversation: () => Promise<string>
  loadConversationHistory: (id: string) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  newChat: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversationId: null,
  messages: [],
  isStreaming: false,
  streamingContent: '',
  isWaitingForResponse: false,

  setCurrentConversation: (id) => set({ currentConversationId: id }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  setStreaming: (streaming) => set({ isStreaming: streaming }),
  setStreamingContent: (content) => set({ streamingContent: content }),
  appendStreamingContent: (chunk) =>
    set((state) => ({ streamingContent: state.streamingContent + chunk })),
  setWaitingForResponse: (waiting) => set({ isWaitingForResponse: waiting }),

  loadConversations: async () => {
    try {
      const data = await api<{ conversations: Conversation[] }>(
        '/conversations',
      )
      set({ conversations: data.conversations || [] })
    } catch {
      console.error('Failed to load conversations')
    }
  },

  createConversation: async () => {
    const data = await api<Conversation>('/conversations', {
      method: 'POST',
      body: JSON.stringify({ title: 'New Conversation' }),
    })
    set({ currentConversationId: data.id, messages: [] })
    await get().loadConversations()
    return data.id
  },

  loadConversationHistory: async (id: string) => {
    try {
      const data = await api<Conversation>(`/conversations/${id}`)
      set({
        currentConversationId: id,
        messages: data.messages || [],
      })
    } catch {
      console.error('Failed to load conversation history')
    }
  },

  deleteConversation: async (id: string) => {
    await api(`/conversations/${id}`, { method: 'DELETE' })
    const { currentConversationId } = get()
    if (currentConversationId === id) {
      get().newChat()
    }
    await get().loadConversations()
  },

  newChat: () => {
    set({
      currentConversationId: null,
      messages: [],
      isStreaming: false,
      streamingContent: '',
      isWaitingForResponse: false,
    })
  },
}))
