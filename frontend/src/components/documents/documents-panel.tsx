import { useState, useEffect, useCallback, useRef } from 'react'
import { useClientStore } from '@/stores/client-store'
import { useToast } from '@/components/ui/toast'
import { api, apiUpload } from '@/lib/api'
import { formatFileSize } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { X, Upload, FileText, Trash2, Plus, Loader2 } from 'lucide-react'

interface Document {
  id: string
  filename?: string
  name?: string
  size?: number
  chunk_count?: number
  client_name?: string
}

interface DocumentsPanelProps {
  isOpen: boolean
  onClose: () => void
}

export function DocumentsPanel({ isOpen, onClose }: DocumentsPanelProps) {
  const { clients, currentClientId, setCurrentClient, createClient } = useClientStore()
  const { showToast } = useToast()
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<string | null>(null)
  const [useOcr, setUseOcr] = useState(true)
  const [fastMode, setFastMode] = useState(false)
  const [showClientModal, setShowClientModal] = useState(false)
  const [newClientName, setNewClientName] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadDocuments = useCallback(async () => {
    setIsLoading(true)
    try {
      let url = '/documents'
      if (currentClientId) url += `?client_id=${currentClientId}`
      const data = await api<Document[]>(url)
      setDocuments(Array.isArray(data) ? data : [])
    } catch {
      setDocuments([])
    } finally {
      setIsLoading(false)
    }
  }, [currentClientId])

  useEffect(() => {
    if (isOpen) loadDocuments()
  }, [isOpen, loadDocuments])

  const handleUpload = async (files: FileList) => {
    for (let i = 0; i < files.length; i++) {
      const file = files[i]!
      setUploadProgress(`Uploading ${file.name} (${i + 1}/${files.length})`)
      const formData = new FormData()
      formData.append('files', file)
      formData.append('use_ocr', String(useOcr))
      formData.append('fast_mode', String(fastMode))
      if (currentClientId) formData.append('client_id', currentClientId)

      try {
        await apiUpload('/documents/upload', formData)
        showToast(`Uploaded: ${file.name}`, 'success')
      } catch (err) {
        showToast(`Failed: ${err instanceof Error ? err.message : file.name}`, 'error')
      }
    }
    setUploadProgress(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    loadDocuments()
  }

  const handleDelete = async (id: string) => {
    try {
      await api(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
      showToast('Document deleted', 'success')
      loadDocuments()
    } catch {
      showToast('Failed to delete', 'error')
    }
  }

  const handleCreateClient = async () => {
    if (!newClientName.trim()) return
    try {
      await createClient(newClientName.trim())
      showToast(`Client "${newClientName}" created`, 'success')
      setShowClientModal(false)
      setNewClientName('')
      loadDocuments()
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to create client', 'error')
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    handleUpload(e.dataTransfer.files)
  }

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-y-0 right-0 z-40 flex w-80 flex-col border-l bg-background shadow-xl animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <h3 className="font-semibold text-sm">Documents</h3>
          <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7">
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Client selector */}
        <div className="flex items-center gap-2 border-b p-3">
          <select
            value={currentClientId}
            onChange={(e) => { setCurrentClient(e.target.value); }}
            className="h-8 flex-1 rounded-md border bg-background px-2 text-xs"
          >
            {clients.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
          <Button variant="outline" size="icon" className="h-8 w-8" onClick={() => setShowClientModal(true)}>
            <Plus className="h-3 w-3" />
          </Button>
        </div>

        {/* Upload zone */}
        <div className="p-3 space-y-3">
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-4 text-center transition-colors hover:border-primary/50 hover:bg-accent/50"
          >
            <Upload className="h-5 w-5 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">
              Drop files here or <span className="text-primary font-medium">browse</span>
            </p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && handleUpload(e.target.files)}
          />

          {/* Upload options */}
          <div className="flex gap-4 text-xs">
            <label className="flex items-center gap-1.5 text-muted-foreground">
              <input type="checkbox" checked={useOcr} onChange={(e) => setUseOcr(e.target.checked)} className="accent-primary" />
              OCR
            </label>
            <label className="flex items-center gap-1.5 text-muted-foreground">
              <input type="checkbox" checked={fastMode} onChange={(e) => setFastMode(e.target.checked)} className="accent-primary" />
              Fast Mode
            </label>
          </div>

          {uploadProgress && (
            <div className="flex items-center gap-2 rounded-md bg-muted p-2 text-xs">
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>{uploadProgress}</span>
            </div>
          )}
        </div>

        {/* Document list */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {isLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : documents.length === 0 ? (
            <p className="py-8 text-center text-xs text-muted-foreground">No documents uploaded</p>
          ) : (
            documents.map((doc) => (
              <div key={doc.id} className="flex items-center gap-3 rounded-md border bg-card p-3">
                <FileText className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium">{doc.filename || doc.name || 'Document'}</p>
                  <p className="text-xs text-muted-foreground">
                    {doc.chunk_count ? `${doc.chunk_count} chunks` : formatFileSize(doc.size || 0)}
                    {doc.client_name && ` · ${doc.client_name}`}
                  </p>
                </div>
                <button onClick={() => handleDelete(doc.id)} className="text-muted-foreground hover:text-destructive">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Create client modal */}
      <Dialog open={showClientModal} onOpenChange={setShowClientModal}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>Create New Client</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Client name"
            value={newClientName}
            onChange={(e) => setNewClientName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateClient()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowClientModal(false)}>Cancel</Button>
            <Button onClick={handleCreateClient}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
