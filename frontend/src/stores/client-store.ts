import { create } from 'zustand'
import { api } from '@/lib/api'

export interface Client {
  id: string
  name: string
  aliases?: string[]
  created_at?: string
  metadata?: Record<string, unknown>
}

interface ClientState {
  clients: Client[]
  currentClientId: string
  isLoading: boolean

  setCurrentClient: (id: string) => void
  loadClients: () => Promise<void>
  createClient: (name: string, aliases?: string[]) => Promise<Client>
}

export const useClientStore = create<ClientState>((set, get) => ({
  clients: [],
  currentClientId: localStorage.getItem('current_client_id') || '',
  isLoading: false,

  setCurrentClient: (id: string) => {
    localStorage.setItem('current_client_id', id)
    set({ currentClientId: id })
  },

  loadClients: async () => {
    set({ isLoading: true })
    try {
      let data: { clients: Client[] }
      try {
        data = await api<{ clients: Client[] }>('/clients/my/assigned')
      } catch {
        data = await api<{ clients: Client[] }>('/clients')
      }

      const clients = data.clients || []
      const { currentClientId } = get()

      const storedExists = clients.some((c) => c.id === currentClientId)
      if (!storedExists && clients.length > 0) {
        const firstId = clients[0]!.id
        localStorage.setItem('current_client_id', firstId)
        set({ clients, currentClientId: firstId, isLoading: false })
      } else {
        set({ clients, isLoading: false })
      }
    } catch {
      set({ isLoading: false })
    }
  },

  createClient: async (name: string, aliases?: string[]) => {
    const client = await api<Client>('/clients', {
      method: 'POST',
      body: JSON.stringify({ name, aliases: aliases || [], metadata: {} }),
    })
    await get().loadClients()
    set({ currentClientId: client.id })
    localStorage.setItem('current_client_id', client.id)
    return client
  },
}))
