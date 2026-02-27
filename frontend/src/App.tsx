import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ToastProvider } from '@/components/ui/toast'
import { ProtectedRoute } from '@/components/auth/protected-route'
import { LoginPage } from '@/pages/login-page'
import { ChatPage } from '@/pages/chat-page'
import { ProfilePage } from '@/pages/profile-page'

export function App() {
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark')
    }
  }, [])

  return (
    <ToastProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <ChatPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ToastProvider>
  )
}
