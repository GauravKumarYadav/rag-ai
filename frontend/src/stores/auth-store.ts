import { create } from 'zustand'
import { api } from '@/lib/api'

export interface User {
  id: string
  username: string
  email?: string
  is_superuser: boolean
  is_active: boolean
  created_at?: string
  allowed_clients?: string[]
}

interface AuthState {
  token: string | null
  user: User | null
  isLoading: boolean
  setToken: (token: string) => void
  setUser: (user: User) => void
  logout: () => void
  checkAuth: () => Promise<boolean>
  login: (username: string, password: string) => Promise<void>
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem('auth_token'),
  user: (() => {
    try {
      const stored = localStorage.getItem('user')
      return stored ? (JSON.parse(stored) as User) : null
    } catch {
      return null
    }
  })(),
  isLoading: false,

  setToken: (token: string) => {
    localStorage.setItem('auth_token', token)
    set({ token })
  },

  setUser: (user: User) => {
    localStorage.setItem('user', JSON.stringify(user))
    set({ user })
  },

  logout: () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('user')
    set({ token: null, user: null })
  },

  checkAuth: async () => {
    const { token } = get()
    if (!token) return false

    try {
      const user = await api<User>('/auth/me')
      set({ user })
      localStorage.setItem('user', JSON.stringify(user))
      return true
    } catch {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('user')
      set({ token: null, user: null })
      return false
    }
  },

  login: async (username: string, password: string) => {
    set({ isLoading: true })
    try {
      const data = await api<{ access_token: string }>('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
        skipAuth: true,
      })

      localStorage.setItem('auth_token', data.access_token)
      set({ token: data.access_token })

      const user = await api<User>('/auth/me')
      localStorage.setItem('user', JSON.stringify(user))
      set({ user, isLoading: false })
    } catch (error) {
      set({ isLoading: false })
      throw error
    }
  },
}))
