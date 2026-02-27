import { useEffect, useState } from 'react'
import { useClientStore } from '@/stores/client-store'
import { useChatStore } from '@/stores/chat-store'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { Menu, FileText, Sun, Moon } from 'lucide-react'

interface HeaderProps {
  onToggleSidebar: () => void
  onToggleDocuments: () => void
}

export function Header({ onToggleSidebar, onToggleDocuments }: HeaderProps) {
  const { clients, currentClientId, setCurrentClient } = useClientStore()
  const { currentConversationId } = useChatStore()
  const [modelName, setModelName] = useState('AI Model')
  const [isConnected, setIsConnected] = useState(false)
  const [isDark, setIsDark] = useState(() => document.documentElement.classList.contains('dark'))

  useEffect(() => {
    fetch('/status')
      .then((r) => r.json())
      .then((data: { model?: string }) => {
        if (data.model) setModelName(data.model)
        setIsConnected(true)
      })
      .catch(() => setIsConnected(false))
  }, [])

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/health')
        setIsConnected(res.ok)
      } catch {
        setIsConnected(false)
      }
    }, 120000)
    return () => clearInterval(interval)
  }, [])

  const toggleTheme = () => {
    const next = !isDark
    setIsDark(next)
    document.documentElement.classList.toggle('dark', next)
    localStorage.setItem('theme', next ? 'dark' : 'light')
  }

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/80 px-4 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={onToggleSidebar} className="h-8 w-8">
          <Menu className="h-4 w-4" />
        </Button>
        <h3 className="truncate text-sm font-semibold max-w-[200px]">
          {currentConversationId ? 'Conversation' : 'New Conversation'}
        </h3>
        <span className="hidden rounded-md bg-secondary px-2 py-0.5 text-xs text-muted-foreground sm:inline-block">
          {modelName}
        </span>
      </div>

      <div className="flex items-center gap-2">
        {/* Connection status */}
        <div className="flex items-center gap-1.5">
          <div className={cn('h-2 w-2 rounded-full', isConnected ? 'bg-green-500' : 'bg-red-500')} />
          <span className="hidden text-xs text-muted-foreground sm:inline">{isConnected ? 'Connected' : 'Offline'}</span>
        </div>

        {/* Client selector */}
        {clients.length > 0 && (
          <select
            value={currentClientId}
            onChange={(e) => setCurrentClient(e.target.value)}
            className="h-8 rounded-md border bg-background px-2 text-xs"
          >
            {clients.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        )}

        {/* Theme toggle */}
        <Button variant="ghost" size="icon" onClick={toggleTheme} className="h-8 w-8">
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        {/* Documents button */}
        <Button variant="outline" size="sm" onClick={onToggleDocuments} className="h-8">
          <FileText className="h-4 w-4" />
          <span className="hidden sm:inline">Docs</span>
        </Button>
      </div>
    </header>
  )
}
