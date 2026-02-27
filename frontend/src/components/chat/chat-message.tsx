import { cn } from '@/lib/utils'
import { MarkdownRenderer } from './markdown-renderer'
import { Bot, User } from 'lucide-react'

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: string
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  return (
    <div className={cn('flex gap-3 max-w-[900px] mx-auto w-full', role === 'user' && 'flex-row-reverse')}>
      <div
        className={cn(
          'flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg',
          role === 'assistant' ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground',
        )}
      >
        {role === 'assistant' ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
      </div>
      <div
        className={cn(
          'rounded-lg border px-4 py-3 text-sm leading-relaxed max-w-[85%] min-w-0',
          role === 'user'
            ? 'bg-primary text-primary-foreground border-transparent'
            : 'bg-card',
        )}
      >
        {role === 'assistant' ? (
          <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <MarkdownRenderer content={content} />
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{content}</p>
        )}
      </div>
    </div>
  )
}
