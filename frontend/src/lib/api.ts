import { getApiBase } from './utils'

const API_BASE = getApiBase()

interface RequestOptions extends RequestInit {
  skipAuth?: boolean
}

function getToken(): string | null {
  return localStorage.getItem('auth_token')
}

function clearAuth() {
  localStorage.removeItem('auth_token')
  localStorage.removeItem('user')
  window.location.href = '/login'
}

export async function api<T = unknown>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { skipAuth = false, headers: customHeaders, ...rest } = options

  const headers: Record<string, string> = {
    ...(customHeaders as Record<string, string>),
  }

  if (!skipAuth) {
    const token = getToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
  }

  if (
    rest.body &&
    typeof rest.body === 'string' &&
    !headers['Content-Type']
  ) {
    headers['Content-Type'] = 'application/json'
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers,
  })

  if (response.status === 401) {
    clearAuth()
    throw new Error('Unauthorized')
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(
      (errorData as { detail?: string }).detail ||
        `Request failed with status ${response.status}`,
    )
  }

  const contentType = response.headers.get('content-type')
  if (contentType?.includes('application/json')) {
    return response.json() as Promise<T>
  }

  return response as unknown as T
}

export async function apiUpload<T = unknown>(
  path: string,
  formData: FormData,
): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {}

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers,
    body: formData,
  })

  if (response.status === 401) {
    clearAuth()
    throw new Error('Unauthorized')
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(
      (errorData as { detail?: string }).detail ||
        `Upload failed with status ${response.status}`,
    )
  }

  return response.json() as Promise<T>
}
