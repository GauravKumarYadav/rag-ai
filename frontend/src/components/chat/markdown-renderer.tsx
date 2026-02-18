import { useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Copy, Check } from 'lucide-react'
import { useState } from 'react'

function CopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [code])

  return (
    <button
      onClick={handleCopy}
      className="absolute right-2 top-2 rounded-md border border-white/20 bg-white/10 p-1.5 text-xs opacity-0 transition-opacity group-hover:opacity-100 hover:bg-white/20"
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </button>
  )
}

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        pre({ children, ...props }) {
          const codeElement = (children as React.ReactElement[])?.[0]
          const codeString =
            typeof codeElement === 'object' && codeElement?.props?.children
              ? String(codeElement.props.children).replace(/\n$/, '')
              : ''

          return (
            <div className="group relative">
              <pre
                className="overflow-x-auto rounded-lg border bg-[#1e1e2e] p-4 text-sm dark:border-zinc-700"
                {...props}
              >
                {children}
              </pre>
              {codeString && <CopyButton code={codeString} />}
            </div>
          )
        },
        code({ className, children, ...props }) {
          const isInline = !className
          if (isInline) {
            return (
              <code
                className="rounded-sm border bg-muted px-1.5 py-0.5 text-[0.85em] font-mono"
                {...props}
              >
                {children}
              </code>
            )
          }
          return (
            <code className={className} {...props}>
              {children}
            </code>
          )
        },
        table({ children, ...props }) {
          return (
            <div className="my-4 overflow-x-auto">
              <table className="w-full border-collapse border text-sm" {...props}>
                {children}
              </table>
            </div>
          )
        },
        th({ children, ...props }) {
          return (
            <th className="border bg-muted px-3 py-2 text-left font-semibold" {...props}>
              {children}
            </th>
          )
        },
        td({ children, ...props }) {
          return (
            <td className="border px-3 py-2" {...props}>
              {children}
            </td>
          )
        },
        blockquote({ children, ...props }) {
          return (
            <blockquote className="my-3 border-l-4 border-primary/30 bg-muted/50 py-1 pl-4 italic" {...props}>
              {children}
            </blockquote>
          )
        },
        a({ children, href, ...props }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-2 hover:text-primary/80"
              {...props}
            >
              {children}
            </a>
          )
        },
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
