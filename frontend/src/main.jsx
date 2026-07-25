import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App'
import ChatPage from './pages/ChatPage'
import UploadPage from './pages/UploadPage'
import DocumentsPage from './pages/DocumentsPage'
import ConsolePage from './pages/ConsolePage'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<ChatPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="console" element={<ConsolePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
