import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/stores/auth-store'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useToast } from '@/components/ui/toast'
import { ArrowLeft, User, Building, ListChecks, Shield, FlaskConical, BarChart3, LogOut } from 'lucide-react'
import { cn, getInitials } from '@/lib/utils'

const allTabs = [
  { id: 'profile', label: 'Profile', icon: User, adminOnly: false },
  { id: 'clients', label: 'Clients', icon: Building, adminOnly: false },
  { id: 'audit', label: 'Audit Logs', icon: ListChecks, adminOnly: true },
  { id: 'users', label: 'Users', icon: Shield, adminOnly: true },
  { id: 'evaluation', label: 'Evaluation', icon: FlaskConical, adminOnly: true },
  { id: 'stats', label: 'Stats', icon: BarChart3, adminOnly: true },
] as const

type TabId = (typeof allTabs)[number]['id']

export function ProfilePage() {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [currentTab, setCurrentTab] = useState<TabId>('profile')

  const visibleTabs = user?.is_superuser ? allTabs : allTabs.filter((t) => !t.adminOnly)

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-[260px] flex-shrink-0 flex-col border-r bg-sidebar">
        <div className="flex items-center gap-2 border-b p-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-xs text-primary-foreground">
            <User className="h-4 w-4" />
          </div>
          <span className="font-semibold text-sm">Settings</span>
        </div>
        <nav className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {visibleTabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setCurrentTab(tab.id)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  currentTab === tab.id
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                )}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </nav>
        <div className="border-t p-3">
          <Button variant="ghost" size="sm" className="w-full justify-start text-destructive hover:text-destructive" onClick={handleLogout}>
            <LogOut className="h-4 w-4" /> Logout
          </Button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex flex-1 flex-col min-w-0">
        <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/80 px-6 backdrop-blur-sm">
          <h3 className="font-semibold">{visibleTabs.find((t) => t.id === currentTab)?.label}</h3>
          <div className="flex items-center gap-3">
            <span className="rounded-md bg-secondary px-3 py-1 text-xs font-medium">
              {user?.username} ({user?.is_superuser ? 'Admin' : 'User'})
            </span>
            <Button variant="outline" size="sm" onClick={() => navigate('/')}>
              <ArrowLeft className="h-4 w-4" /> Back to Chat
            </Button>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-6">
          {currentTab === 'profile' && <ProfileTab />}
          {currentTab === 'clients' && <ClientsTab />}
          {currentTab === 'audit' && <AuditTab />}
          {currentTab === 'users' && <UsersTab />}
          {currentTab === 'evaluation' && <EvaluationTab />}
          {currentTab === 'stats' && <StatsTab />}
        </div>
      </main>
    </div>
  )
}

// ===== Profile Tab =====
function ProfileTab() {
  const { user } = useAuthStore()
  const { showToast } = useToast()
  const [currentPass, setCurrentPass] = useState('')
  const [newPass, setNewPass] = useState('')
  const [confirmPass, setConfirmPass] = useState('')

  const handleChangePassword = async () => {
    if (!currentPass || !newPass) { showToast('Please fill in all fields', 'error'); return }
    if (newPass !== confirmPass) { showToast('Passwords do not match', 'error'); return }
    if (newPass.length < 6) { showToast('Password must be at least 6 characters', 'error'); return }
    try {
      await api('/auth/change-password', { method: 'POST', body: JSON.stringify({ current_password: currentPass, new_password: newPass }) })
      showToast('Password updated', 'success')
      setCurrentPass(''); setNewPass(''); setConfirmPass('')
    } catch (err) { showToast(err instanceof Error ? err.message : 'Failed', 'error') }
  }

  if (!user) return null
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center gap-6 rounded-lg border bg-card p-6">
        <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary text-3xl font-bold text-primary-foreground">
          {getInitials(user.username)}
        </div>
        <div>
          <h2 className="text-xl font-bold">{user.username}</h2>
          <p className="text-sm text-muted-foreground">{user.email || 'No email set'}</p>
          <span className="mt-1 inline-block rounded bg-secondary px-2 py-0.5 text-xs font-medium">
            {user.is_superuser ? 'Administrator' : 'User'}
          </span>
        </div>
      </div>
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Change Password</h3>
        <div className="space-y-4">
          <Input type="password" placeholder="Current password" value={currentPass} onChange={(e) => setCurrentPass(e.target.value)} />
          <div className="grid grid-cols-2 gap-4">
            <Input type="password" placeholder="New password" value={newPass} onChange={(e) => setNewPass(e.target.value)} />
            <Input type="password" placeholder="Confirm new password" value={confirmPass} onChange={(e) => setConfirmPass(e.target.value)} />
          </div>
          <Button onClick={handleChangePassword}>Update Password</Button>
        </div>
      </div>
    </div>
  )
}

// ===== Clients Tab =====
function ClientsTab() {
  const { showToast } = useToast()
  const { user } = useAuthStore()
  const [clients, setClients] = useState<Array<{ id: string; name: string; aliases?: string[]; created_at?: string }>>([])
  const [name, setName] = useState('')
  const [aliases, setAliases] = useState('')

  const load = useCallback(async () => {
    try {
      const data = await api<{ clients: typeof clients }>('/clients')
      setClients(data.clients || [])
    } catch { /* */ }
  }, [])

  useEffect(() => { load() }, [load])

  const handleCreate = async () => {
    if (!name.trim()) return
    try {
      await api('/clients', { method: 'POST', body: JSON.stringify({ name: name.trim(), aliases: aliases ? aliases.split(',').map((a) => a.trim()).filter(Boolean) : [], metadata: {} }) })
      showToast('Client created', 'success')
      setName(''); setAliases(''); load()
    } catch (err) { showToast(err instanceof Error ? err.message : 'Failed', 'error') }
  }

  const handleDelete = async (id: string, cname: string) => {
    if (!confirm(`Delete client "${cname}"?`)) return
    try {
      await api(`/clients/${id}`, { method: 'DELETE' })
      load()
    } catch (err) { showToast(err instanceof Error ? err.message : 'Failed', 'error') }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Clients ({clients.length})</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">ID</th><th className="p-3 text-left font-medium">Name</th><th className="p-3 text-left font-medium">Aliases</th><th className="p-3 text-left font-medium">Created</th>{user?.is_superuser && <th className="p-3 text-left font-medium">Actions</th>}</tr></thead>
            <tbody>
              {clients.map((c) => (
                <tr key={c.id} className="border-b hover:bg-muted/30">
                  <td className="p-3 font-mono text-xs text-muted-foreground">{c.id.substring(0, 8)}...</td>
                  <td className="p-3 font-medium">{c.name}</td>
                  <td className="p-3 text-muted-foreground">{c.aliases?.join(', ') || '-'}</td>
                  <td className="p-3 text-muted-foreground">{c.created_at ? new Date(c.created_at).toLocaleDateString() : '-'}</td>
                  {user?.is_superuser && <td className="p-3"><Button variant="ghost" size="sm" className="text-destructive" onClick={() => handleDelete(c.id, c.name)}>Delete</Button></td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {user?.is_superuser && (
        <div className="rounded-lg border bg-card p-6">
          <h3 className="text-lg font-semibold mb-4">Create Client</h3>
          <div className="grid grid-cols-2 gap-4">
            <Input placeholder="Client name" value={name} onChange={(e) => setName(e.target.value)} />
            <Input placeholder="Aliases (comma-separated)" value={aliases} onChange={(e) => setAliases(e.target.value)} />
          </div>
          <Button className="mt-4" onClick={handleCreate}>Create Client</Button>
        </div>
      )}
    </div>
  )
}

// ===== Audit Tab =====
function AuditTab() {
  const [logs, setLogs] = useState<Array<{ username?: string; action: string; method?: string; path?: string; status_code?: number; timestamp: string }>>([])
  const [total, setTotal] = useState(0)

  const load = useCallback(async () => {
    try {
      const data = await api<{ logs: typeof logs; total: number }>('/admin/audit-logs?page=1&page_size=50')
      setLogs(data.logs || []); setTotal(data.total || 0)
    } catch { /* */ }
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <div className="rounded-lg border bg-card p-6">
      <h3 className="text-lg font-semibold mb-4">Recent Logs ({total})</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">User</th><th className="p-3 text-left font-medium">Action</th><th className="p-3 text-left font-medium">Path</th><th className="p-3 text-left font-medium">Status</th><th className="p-3 text-left font-medium">Time</th></tr></thead>
          <tbody>
            {logs.map((log, i) => (
              <tr key={i} className="border-b hover:bg-muted/30">
                <td className="p-3">{log.username || '-'}</td>
                <td className="p-3"><span className="rounded bg-secondary px-2 py-0.5 text-xs font-mono">{log.action}</span></td>
                <td className="p-3 font-mono text-xs text-muted-foreground">{log.method} {log.path}</td>
                <td className="p-3"><span className={log.status_code && log.status_code >= 400 ? 'text-destructive' : 'text-green-600'}>{log.status_code}</span></td>
                <td className="p-3 text-muted-foreground">{new Date(log.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ===== Users Tab =====
function UsersTab() {
  const { showToast } = useToast()
  const [users, setUsers] = useState<Array<{ id: string; username: string; email?: string; is_superuser: boolean; is_active: boolean; created_at?: string }>>([])
  const [newUser, setNewUser] = useState('')
  const [newEmail, setNewEmail] = useState('')
  const [newPass, setNewPass] = useState('')
  const [newRole, setNewRole] = useState('false')

  const load = useCallback(async () => {
    try {
      const data = await api<{ users: typeof users }>('/admin/users?limit=100')
      setUsers(data.users || [])
    } catch { /* */ }
  }, [])

  useEffect(() => { load() }, [load])

  const handleCreate = async () => {
    try {
      await api('/admin/users', { method: 'POST', body: JSON.stringify({ username: newUser, email: newEmail, password: newPass, is_superuser: newRole === 'true' }) })
      showToast('User created', 'success')
      setNewUser(''); setNewEmail(''); setNewPass(''); load()
    } catch (err) { showToast(err instanceof Error ? err.message : 'Failed', 'error') }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Users ({users.length})</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Username</th><th className="p-3 text-left font-medium">Email</th><th className="p-3 text-left font-medium">Role</th><th className="p-3 text-left font-medium">Status</th><th className="p-3 text-left font-medium">Created</th></tr></thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-b hover:bg-muted/30">
                  <td className="p-3 font-medium">{u.username}</td>
                  <td className="p-3 text-muted-foreground">{u.email || '-'}</td>
                  <td className="p-3"><span className={cn('rounded px-2 py-0.5 text-xs', u.is_superuser ? 'bg-primary text-primary-foreground' : 'bg-secondary')}>{u.is_superuser ? 'Admin' : 'User'}</span></td>
                  <td className="p-3"><span className={u.is_active ? 'text-green-600' : 'text-destructive'}>{u.is_active ? 'Active' : 'Disabled'}</span></td>
                  <td className="p-3 text-muted-foreground">{u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Create User</h3>
        <div className="grid grid-cols-4 gap-4">
          <Input placeholder="Username" value={newUser} onChange={(e) => setNewUser(e.target.value)} />
          <Input placeholder="Email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
          <Input type="password" placeholder="Password" value={newPass} onChange={(e) => setNewPass(e.target.value)} />
          <select value={newRole} onChange={(e) => setNewRole(e.target.value)} className="h-10 rounded-md border bg-background px-3 text-sm">
            <option value="false">User</option>
            <option value="true">Admin</option>
          </select>
        </div>
        <Button className="mt-4" onClick={handleCreate}>Create User</Button>
      </div>
    </div>
  )
}

// ===== Evaluation Tab =====
function EvaluationTab() {
  const { showToast } = useToast()
  const [datasets, setDatasets] = useState<Array<{ id: number; name: string; client_id?: string; sample_size: number; status?: string; created_at?: string }>>([])
  const [runs, setRuns] = useState<Array<{ id: number; dataset_name?: string; dataset_id: number; status?: string; metrics?: Record<string, number>; created_at?: string }>>([])

  const load = useCallback(async () => {
    try {
      const ds = await api<{ datasets: typeof datasets }>('/evaluation/datasets')
      setDatasets(ds.datasets || [])
    } catch { /* */ }
    try {
      const r = await api<{ runs: typeof runs }>('/evaluation/runs')
      setRuns(r.runs || [])
    } catch { /* */ }
  }, [])

  useEffect(() => { load() }, [load])

  const [dsName, setDsName] = useState('')
  const [dsClient, setDsClient] = useState('')
  const [dsSize, setDsSize] = useState('')

  const handleGenerate = async () => {
    try {
      await api('/evaluation/datasets', { method: 'POST', body: JSON.stringify({ name: dsName, client_id: dsClient || null, sample_size: parseInt(dsSize) || null }) })
      showToast('Dataset created', 'success'); load()
    } catch (err) { showToast(err instanceof Error ? err.message : 'Failed', 'error') }
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Datasets</h3>
        <table className="w-full text-sm">
          <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">ID</th><th className="p-3 text-left font-medium">Name</th><th className="p-3 text-left font-medium">Client</th><th className="p-3 text-left font-medium">Sample</th><th className="p-3 text-left font-medium">Status</th><th className="p-3 text-left font-medium">Created</th></tr></thead>
          <tbody>{datasets.map((d) => <tr key={d.id} className="border-b"><td className="p-3">{d.id}</td><td className="p-3 font-medium">{d.name}</td><td className="p-3 text-muted-foreground">{d.client_id || '-'}</td><td className="p-3">{d.sample_size}</td><td className="p-3"><span className="rounded bg-secondary px-2 py-0.5 text-xs">{d.status || 'ready'}</span></td><td className="p-3 text-muted-foreground">{d.created_at ? new Date(d.created_at).toLocaleDateString() : '-'}</td></tr>)}</tbody>
        </table>
      </div>
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Runs</h3>
        <table className="w-full text-sm">
          <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">ID</th><th className="p-3 text-left font-medium">Dataset</th><th className="p-3 text-left font-medium">Status</th><th className="p-3 text-left font-medium">Precision</th><th className="p-3 text-left font-medium">Recall</th><th className="p-3 text-left font-medium">F1</th><th className="p-3 text-left font-medium">MRR</th><th className="p-3 text-left font-medium">NDCG</th></tr></thead>
          <tbody>{runs.map((r) => { const m = r.metrics || {}; return <tr key={r.id} className="border-b"><td className="p-3">{r.id}</td><td className="p-3">{r.dataset_name || r.dataset_id}</td><td className="p-3"><span className="rounded bg-secondary px-2 py-0.5 text-xs">{r.status || 'completed'}</span></td><td className="p-3 font-mono">{m['precision']?.toFixed(3) ?? '-'}</td><td className="p-3 font-mono">{m['recall']?.toFixed(3) ?? '-'}</td><td className="p-3 font-mono">{m['f1']?.toFixed(3) ?? '-'}</td><td className="p-3 font-mono">{m['mrr']?.toFixed(3) ?? '-'}</td><td className="p-3 font-mono">{m['ndcg']?.toFixed(3) ?? '-'}</td></tr> })}</tbody>
        </table>
      </div>
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Generate Dataset</h3>
        <div className="grid grid-cols-3 gap-4">
          <Input placeholder="Name" value={dsName} onChange={(e) => setDsName(e.target.value)} />
          <Input placeholder="Client ID (optional)" value={dsClient} onChange={(e) => setDsClient(e.target.value)} />
          <Input type="number" placeholder="Sample size" value={dsSize} onChange={(e) => setDsSize(e.target.value)} />
        </div>
        <Button className="mt-4" onClick={handleGenerate}>Generate</Button>
      </div>
    </div>
  )
}

// ===== Stats Tab =====
interface AdminStats {
  documents?: { total_chunks?: number; memories?: number }
  sessions?: { active_conversations?: number }
  memory?: { max_context_tokens?: number; summary_target_tokens?: number; sliding_window_size?: number }
}

interface AdminConfig {
  llm_provider?: string
  llm_model?: string
  embedding_model?: string
  rag_provider?: string
  agent_enabled?: boolean
  bm25_enabled?: boolean
  reranker_enabled?: boolean
  log_level?: string
}

function StatsTab() {
  const [stats, setStats] = useState<AdminStats>({})
  const [config, setConfig] = useState<AdminConfig>({})

  useEffect(() => {
    api<AdminStats>('/admin/stats').then(setStats).catch(() => {})
    api<AdminConfig>('/admin/config').then(setConfig).catch(() => {})
  }, [])

  const statCards = [
    { label: 'Doc Chunks', value: stats.documents?.total_chunks },
    { label: 'Memories', value: stats.documents?.memories },
    { label: 'Conversations', value: stats.sessions?.active_conversations },
    { label: 'Max Context', value: stats.memory?.max_context_tokens },
    { label: 'Summary Target', value: stats.memory?.summary_target_tokens },
    { label: 'Window Size', value: stats.memory?.sliding_window_size },
  ]

  const configEntries: [string, string][] = Object.entries(config).map(([k, v]) => [k, String(v)])

  return (
    <div className="space-y-6">
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">System Stats</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {statCards.map((s) => (
            <div key={s.label} className="rounded-lg border bg-muted/30 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{s.label}</p>
              <p className="mt-1 text-2xl font-bold">{s.value != null ? String(s.value) : '-'}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-lg border bg-card p-6">
        <h3 className="text-lg font-semibold mb-4">Configuration</h3>
        <table className="w-full text-sm">
          <tbody>
            {configEntries.map(([k, v]) => (
              <tr key={k} className="border-b"><td className="p-3 font-mono text-muted-foreground w-1/3">{k}</td><td className="p-3 font-mono">{v}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
