import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useChatStore } from '@/stores/chat-store'
import { useAuthStore } from '@/stores/auth-store'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { Plus, Trash2, MessageSquare, LogOut, Settings, User } from 'lucide-react'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
}

export function Sidebar({ isOpen }: SidebarProps) {
  const { conversations, currentConversationId, loadConversations, deleteConversation, newChat, setCurrentConversation, loadConversationHistory } = useChatStore()
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  const handleSelectConversation = (id: string) => {
    if (id === currentConversationId) return
    setCurrentConversation(id)
    loadConversationHistory(id)
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (!confirm('Delete this conversation?')) return
    await deleteConversation(id)
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside
      className={cn(
        'flex h-full w-[260px] flex-shrink-0 flex-col border-r bg-sidebar transition-all duration-300',
        'max-md:fixed max-md:left-0 max-md:top-0 max-md:bottom-0 max-md:z-30 max-md:shadow-lg',
        isOpen ? 'max-md:translate-x-0' : 'max-md:-translate-x-full',
        !isOpen && 'md:-ml-[260px]',
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-xs text-primary-foreground font-semibold">
            AI
          </div>
          <span className="font-semibold text-sm">RAG Chat</span>
        </div>
        <Button variant="ghost" size="icon" onClick={newChat} title="New chat">
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 ? (
          <p className="px-3 py-8 text-center text-xs text-muted-foreground">No conversations yet</p>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => handleSelectConversation(conv.id)}
              className={cn(
                'group flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors',
                currentConversationId === conv.id
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
              )}
            >
              <MessageSquare className="h-4 w-4 flex-shrink-0" />
              <span className="flex-1 truncate">{conv.title || 'Untitled'}</span>
              <button
                onClick={(e) => handleDelete(e, conv.id)}
                className="hidden rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive group-hover:block"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="border-t p-3 space-y-1">
        <Button variant="ghost" size="sm" className="w-full justify-start" onClick={() => navigate('/profile')}>
          <User className="h-4 w-4" />
          <span className="truncate">{user?.username || 'User'}</span>
          {user?.is_superuser && (
            <span className="ml-auto rounded bg-primary px-1.5 py-0.5 text-[10px] text-primary-foreground">
              Admin
            </span>
          )}
        </Button>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" className="flex-1 justify-start" onClick={() => navigate('/profile')}>
            <Settings className="h-4 w-4" />
            Settings
          </Button>
          <Button variant="ghost" size="sm" onClick={handleLogout} className="text-destructive hover:text-destructive">
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </aside>
  )
}
