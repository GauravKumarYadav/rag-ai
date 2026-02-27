import { useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { SendHorizontal } from 'lucide-react'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleResize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 150) + 'px'
  }, [])

  const handleSend = useCallback(() => {
    const value = textareaRef.current?.value.trim()
    if (!value) return
    onSend(value)
    if (textareaRef.current) {
      textareaRef.current.value = ''
      textareaRef.current.style.height = 'auto'
    }
  }, [onSend])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend],
  )

  return (
    <div className="border-t bg-background p-4">
      <div className="mx-auto flex max-w-[900px] items-end gap-2 rounded-lg border bg-card p-2 shadow-sm transition-colors focus-within:border-ring focus-within:shadow-md">
        <textarea
          ref={textareaRef}
          placeholder="Type a message..."
          rows={1}
          disabled={disabled}
          onInput={handleResize}
          onKeyDown={handleKeyDown}
          className="flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
        />
        <Button size="icon" className="h-8 w-8 flex-shrink-0" onClick={handleSend} disabled={disabled}>
          <SendHorizontal className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
